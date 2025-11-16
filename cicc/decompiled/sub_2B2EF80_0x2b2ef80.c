// Function: sub_2B2EF80
// Address: 0x2b2ef80
//
__int64 __fastcall sub_2B2EF80(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 *v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r14
  __int64 v7; // rax
  int v8; // r8d
  __int64 v9; // rdi
  int v10; // r8d
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // r10
  __int64 v14; // r13
  int v16; // eax
  int v17; // r9d
  __int64 *v18; // [rsp+8h] [rbp-38h]

  v18 = &a2[a3];
  if ( a2 == v18 )
    return 0;
  v4 = a2;
  v5 = 0;
  v6 = 0;
  do
  {
    while ( 1 )
    {
      v14 = *v4;
      if ( !sub_2B16010(*v4) || !(unsigned __int8)sub_2B099C0(v14) )
        break;
      if ( v18 == ++v4 )
        return v6;
    }
    v7 = 0;
    if ( *(_BYTE *)v14 > 0x1Cu && *(_QWORD *)a1 == *(_QWORD *)(v14 + 40) )
    {
      v8 = *(_DWORD *)(a1 + 104);
      v9 = *(_QWORD *)(a1 + 88);
      if ( v8 )
      {
        v10 = v8 - 1;
        v11 = v10 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v12 = (__int64 *)(v9 + 16LL * v11);
        v13 = *v12;
        if ( v14 == *v12 )
        {
LABEL_7:
          v7 = v12[1];
          if ( v7 && *(_DWORD *)(v7 + 136) != *(_DWORD *)(a1 + 204) )
            v7 = 0;
        }
        else
        {
          v16 = 1;
          while ( v13 != -4096 )
          {
            v17 = v16 + 1;
            v11 = v10 & (v16 + v11);
            v12 = (__int64 *)(v9 + 16LL * v11);
            v13 = *v12;
            if ( v14 == *v12 )
              goto LABEL_7;
            v16 = v17;
          }
          v7 = 0;
        }
      }
    }
    if ( v5 )
      *(_QWORD *)(v5 + 24) = v7;
    else
      v6 = v7;
    *(_QWORD *)(v7 + 16) = v6;
    v5 = v7;
    ++v4;
  }
  while ( v18 != v4 );
  return v6;
}
