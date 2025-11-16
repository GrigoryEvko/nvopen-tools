// Function: sub_E21AF0
// Address: 0xe21af0
//
int __fastcall sub_E21AF0(__int64 a1, size_t a2, const void *a3)
{
  unsigned __int64 v4; // rax
  __int64 v7; // rcx
  __int64 v8; // rbx
  _QWORD *v9; // rdx
  __int64 v10; // rdx
  unsigned __int64 *v11; // rax
  unsigned __int64 *v12; // rbx
  unsigned __int64 v13; // rdx
  __int64 v15; // [rsp+8h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 192);
  if ( v4 <= 9 )
  {
    if ( v4 )
    {
      v7 = a1 + 8 * v4;
      v8 = a1;
      while ( 1 )
      {
        v4 = *(_QWORD *)(v8 + 112);
        if ( *(_QWORD *)(v4 + 24) == a2 )
        {
          v15 = v7;
          if ( !a2 )
            break;
          LODWORD(v4) = memcmp(a3, *(const void **)(v4 + 32), a2);
          v7 = v15;
          if ( !(_DWORD)v4 )
            break;
        }
        v8 += 8;
        if ( v8 == v7 )
          goto LABEL_9;
      }
    }
    else
    {
LABEL_9:
      v9 = *(_QWORD **)(a1 + 16);
      v4 = (*v9 + v9[1] + 7LL) & 0xFFFFFFFFFFFFFFF8LL;
      v9[1] = v4 - *v9 + 40;
      if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) > *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
      {
        v11 = (unsigned __int64 *)sub_22077B0(32);
        v12 = v11;
        if ( v11 )
        {
          *v11 = 0;
          v11[1] = 0;
          v11[2] = 0;
          v11[3] = 0;
        }
        v4 = sub_2207820(4096);
        v13 = *(_QWORD *)(a1 + 16);
        v12[2] = 4096;
        *v12 = v4;
        v12[3] = v13;
        *(_QWORD *)(a1 + 16) = v12;
        v12[1] = 40;
      }
      if ( !v4 )
      {
        MEMORY[0x18] = a2;
        MEMORY[0x20] = a3;
        BUG();
      }
      *(_QWORD *)(v4 + 24) = 0;
      *(_QWORD *)(v4 + 32) = 0;
      *(_DWORD *)(v4 + 8) = 5;
      *(_QWORD *)(v4 + 16) = 0;
      *(_QWORD *)v4 = &unk_49E0F88;
      *(_QWORD *)(v4 + 24) = a2;
      *(_QWORD *)(v4 + 32) = a3;
      v10 = *(_QWORD *)(a1 + 192);
      *(_QWORD *)(a1 + 192) = v10 + 1;
      *(_QWORD *)(a1 + 8 * v10 + 112) = v4;
    }
  }
  return v4;
}
