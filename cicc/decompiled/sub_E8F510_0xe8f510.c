// Function: sub_E8F510
// Address: 0xe8f510
//
void __fastcall sub_E8F510(char *a1, char *a2)
{
  char *v2; // r12
  __int64 *v4; // r13
  __int64 *v5; // r15
  __int64 v6; // rax
  unsigned int v7; // edx
  __int64 v8; // rax
  char *v9; // r13
  __int64 v10; // rsi
  __int64 v11; // rcx
  __int64 v12; // rax
  _QWORD *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rax
  unsigned int v16; // [rsp-3Ch] [rbp-3Ch]

  if ( a1 != a2 )
  {
    v2 = a1 + 16;
    if ( a2 != a1 + 16 )
    {
      v4 = *(__int64 **)v2;
      v5 = *(__int64 **)a1;
      v6 = **(_QWORD **)v2;
      if ( !v6 )
        goto LABEL_10;
LABEL_4:
      v7 = *(_DWORD *)(*(_QWORD *)(v6 + 8) + 36LL);
      v8 = *v5;
      if ( *v5 )
      {
LABEL_5:
        v9 = v2 + 16;
        if ( v7 >= *(_DWORD *)(*(_QWORD *)(v8 + 8) + 36LL) )
          goto LABEL_16;
      }
      else
      {
        while ( 1 )
        {
          v16 = v7;
          if ( (*((_BYTE *)v5 + 9) & 0x70) != 0x20 || *((char *)v5 + 8) < 0 )
            BUG();
          *((_BYTE *)v5 + 8) |= 8u;
          v9 = v2 + 16;
          v15 = sub_E807D0(v5[3]);
          *v5 = (__int64)v15;
          if ( v16 < *(_DWORD *)(v15[1] + 36LL) )
            break;
LABEL_16:
          sub_E8F410(v2);
          if ( a2 == v9 )
            return;
LABEL_9:
          v2 = v9;
          v5 = *(__int64 **)a1;
          v4 = *(__int64 **)v9;
          v6 = *v4;
          if ( *v4 )
            goto LABEL_4;
LABEL_10:
          if ( (*((_BYTE *)v4 + 9) & 0x70) != 0x20 || *((char *)v4 + 8) < 0 )
            BUG();
          *((_BYTE *)v4 + 8) |= 8u;
          v14 = sub_E807D0(v4[3]);
          *v4 = (__int64)v14;
          v7 = *(_DWORD *)(v14[1] + 36LL);
          v8 = *v5;
          if ( *v5 )
            goto LABEL_5;
        }
      }
      v10 = *((_QWORD *)v9 - 2);
      v11 = *((_QWORD *)v9 - 1);
      v12 = (v2 - a1) >> 4;
      if ( v2 - a1 > 0 )
      {
        do
        {
          v13 = (_QWORD *)*((_QWORD *)v2 - 2);
          v2 -= 16;
          *((_QWORD *)v2 + 2) = v13;
          *((_QWORD *)v2 + 3) = *((_QWORD *)v2 + 1);
          --v12;
        }
        while ( v12 );
      }
      *(_QWORD *)a1 = v10;
      *((_QWORD *)a1 + 1) = v11;
      if ( a2 != v9 )
        goto LABEL_9;
    }
  }
}
