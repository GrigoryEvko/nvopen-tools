// Function: sub_35F6BA0
// Address: 0x35f6ba0
//
void __fastcall sub_35F6BA0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  size_t v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdx
  char v11; // al
  void *v12; // rdx
  void *v13; // rdx
  size_t v14; // rdx
  char *v15; // rsi

  if ( a5 )
  {
    v8 = strlen((const char *)a5);
    v9 = *(_QWORD *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8);
    if ( v8 == 5 )
    {
      if ( *(_DWORD *)a5 == 1667330163 && *(_BYTE *)(a5 + 4) == 101 )
      {
        v11 = (unsigned __int8)v9 >> 4;
        if ( v11 == 2 )
        {
          v12 = *(void **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v12 > 0xAu )
          {
            qmemcpy(v12, "shared::cta", 11);
            *(_QWORD *)(a4 + 32) += 11LL;
            return;
          }
          v14 = 11;
          v15 = "shared::cta";
          goto LABEL_19;
        }
        if ( v11 == 3 )
        {
          v13 = *(void **)(a4 + 32);
          if ( *(_QWORD *)(a4 + 24) - (_QWORD)v13 > 0xEu )
          {
            qmemcpy(v13, "shared::cluster", 15);
            *(_QWORD *)(a4 + 32) += 15LL;
            return;
          }
          v14 = 15;
          v15 = "shared::cluster";
          goto LABEL_19;
        }
        goto LABEL_22;
      }
      if ( *(_DWORD *)a5 == 1886348147 && *(_BYTE *)(a5 + 4) == 101 )
      {
        if ( (v9 & 0xF) == 3 )
        {
          v10 = *(_QWORD *)(a4 + 32);
          if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v10) > 6 )
          {
            *(_DWORD *)v10 = 1937075299;
            *(_WORD *)(v10 + 4) = 25972;
            *(_BYTE *)(v10 + 6) = 114;
            *(_QWORD *)(a4 + 32) += 7LL;
            return;
          }
          v14 = 7;
          v15 = "cluster";
LABEL_19:
          sub_CB6200(a4, (unsigned __int8 *)v15, v14);
          return;
        }
LABEL_22:
        BUG();
      }
    }
  }
}
