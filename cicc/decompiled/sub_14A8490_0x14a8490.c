// Function: sub_14A8490
// Address: 0x14a8490
//
void __fastcall sub_14A8490(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  __int64 v8; // rdx
  unsigned __int64 v9; // rax
  _QWORD *v10; // rdx
  unsigned __int64 v11; // [rsp-40h] [rbp-40h]

  if ( a3 )
  {
    v5 = a3;
    while ( 1 )
    {
      v6 = sub_1648700(v5);
      v7 = *(_BYTE *)(v6 + 16);
      if ( v7 <= 0x17u )
      {
LABEL_7:
        if ( a2 )
          *a2 = 1;
        goto LABEL_9;
      }
      if ( v7 != 71 )
        break;
      sub_14A8490(a1, a2, *(_QWORD *)(v6 + 8), a4);
LABEL_9:
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        return;
    }
    if ( v7 == 78 )
    {
      v8 = *(unsigned int *)(a1 + 8);
      v9 = v6 | 4;
      if ( (unsigned int)v8 < *(_DWORD *)(a1 + 12) )
      {
LABEL_13:
        v10 = (_QWORD *)(*(_QWORD *)a1 + 16 * v8);
        *v10 = a4;
        v10[1] = v9;
        ++*(_DWORD *)(a1 + 8);
        goto LABEL_9;
      }
    }
    else
    {
      if ( v7 != 29 )
        goto LABEL_7;
      v8 = *(unsigned int *)(a1 + 8);
      v9 = v6 & 0xFFFFFFFFFFFFFFFBLL;
      if ( (unsigned int)v8 < *(_DWORD *)(a1 + 12) )
        goto LABEL_13;
    }
    v11 = v9;
    sub_16CD150(a1, a1 + 16, 0, 16);
    v8 = *(unsigned int *)(a1 + 8);
    v9 = v11;
    goto LABEL_13;
  }
}
