// Function: sub_37C6950
// Address: 0x37c6950
//
__int64 __fastcall sub_37C6950(__int64 a1, int *a2)
{
  __int64 v4; // rbx
  __int64 v5; // r11
  __int64 v6; // r10
  unsigned int v7; // edi
  unsigned int v8; // edx
  __int64 v9; // r8
  __int64 v10; // rcx
  __int64 v11; // rax
  char v12; // si
  __int64 v13; // rax

  v4 = *(_QWORD *)(a1 + 16);
  if ( v4 )
  {
    v5 = *((_QWORD *)a2 + 2);
    v6 = *((_QWORD *)a2 + 1);
    v7 = *a2;
    while ( 1 )
    {
      v8 = *(_DWORD *)(v4 + 32);
      v9 = *(_QWORD *)(v4 + 48);
      v10 = *(_QWORD *)(v4 + 40);
      if ( v7 < v8 || v7 == v8 && (v10 > v6 || v10 == v6 && v9 > v5) )
      {
        v11 = *(_QWORD *)(v4 + 16);
        v12 = 1;
        if ( !v11 )
          goto LABEL_11;
      }
      else
      {
        v11 = *(_QWORD *)(v4 + 24);
        v12 = 0;
        if ( !v11 )
        {
LABEL_11:
          if ( v12 )
            goto LABEL_12;
LABEL_14:
          if ( v8 < v7 || v8 == v7 && (v6 > v10 || v6 == v10 && v5 > v9) )
            return 0;
          return v4;
        }
      }
      v4 = v11;
    }
  }
  v4 = a1 + 8;
LABEL_12:
  if ( *(_QWORD *)(a1 + 24) != v4 )
  {
    v13 = sub_220EF80(v4);
    v5 = *((_QWORD *)a2 + 2);
    v6 = *((_QWORD *)a2 + 1);
    v7 = *a2;
    v9 = *(_QWORD *)(v13 + 48);
    v10 = *(_QWORD *)(v13 + 40);
    v8 = *(_DWORD *)(v13 + 32);
    v4 = v13;
    goto LABEL_14;
  }
  return 0;
}
