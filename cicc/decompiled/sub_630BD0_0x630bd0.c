// Function: sub_630BD0
// Address: 0x630bd0
//
__int64 __fastcall sub_630BD0(__int64 a1, __int64 *a2, unsigned int a3, _DWORD *a4, __int64 *a5)
{
  __int64 result; // rax
  __int64 *v7; // r15
  __int64 *v10; // r14
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 i; // rdi
  __int64 j; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  __int64 *v22; // [rsp+28h] [rbp-38h] BYREF

  result = 0;
  v21 = 0;
  if ( a2 )
  {
    v7 = a2;
    v10 = &v21;
    while ( 1 )
    {
      while ( 1 )
      {
        v11 = *((_BYTE *)v7 + 144);
        if ( (v11 & 0x10) == 0 )
          break;
        v13 = v7[15];
        v22 = 0;
        *v10 = sub_630BD0(a1, *(_QWORD *)(v13 + 160), a3, a4, &v22);
        if ( !v22 )
          goto LABEL_11;
        v10 = v22;
        *a5 = (__int64)v22;
        v7 = (__int64 *)v7[14];
        if ( !v7 )
          return v21;
      }
      if ( (v11 & 0x40) != 0 )
        goto LABEL_11;
      if ( !*v7 )
        goto LABEL_11;
      v18 = *v7;
      if ( (*(_BYTE *)(*v7 + 81) & 0x20) != 0 || v18 == sub_87EA80() )
        goto LABEL_11;
      *a4 = 1;
      if ( !a3
        && (*((_BYTE *)v7 + 145) & 0x20) == 0
        && (*(_BYTE *)(*(_QWORD *)(v18 + 104) + 28LL) & 0xC) == 0
        && (*(_BYTE *)(a1 + 193) & 2) == 0 )
      {
        v19 = v7[15];
        if ( !(unsigned int)sub_8D32E0(v19) )
        {
          i = v19;
          if ( (*(_BYTE *)(v19 + 140) & 0xFB) == 8 )
          {
            i = v19;
            if ( (sub_8D4C10(v19, dword_4F077C4 != 2) & 1) != 0 )
              goto LABEL_10;
            while ( *(_BYTE *)(i + 140) == 12 )
              i = *(_QWORD *)(i + 160);
          }
          if ( (unsigned int)sub_8D3410(i) )
          {
            if ( !*(_QWORD *)(i + 128) )
              goto LABEL_11;
            for ( i = sub_8D40F0(i); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
              ;
          }
          if ( !(unsigned int)sub_8D3AD0() )
            goto LABEL_11;
          for ( j = i; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v16 = *(_QWORD *)(*(_QWORD *)j + 96LL);
          if ( (*(_BYTE *)(v16 + 176) & 1) != 0 )
            goto LABEL_10;
          v17 = *(_QWORD *)(v16 + 16);
          if ( v17 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v17 + 88) + 206LL) & 8) != 0 )
              goto LABEL_35;
          }
          else if ( !*(_QWORD *)(v16 + 8) )
          {
LABEL_35:
            if ( (!dword_4D048B8 || !*(_QWORD *)(v16 + 24) || (*(_BYTE *)(v16 + 177) & 2) != 0)
              && (*(char *)(v16 + 178) >= 0 || (*(_BYTE *)(i + 176) & 2) == 0) )
            {
              goto LABEL_11;
            }
          }
        }
      }
LABEL_10:
      v12 = sub_726BB0(2);
      *(_BYTE *)(v12 + 9) |= 1u;
      *(_QWORD *)(v12 + 16) = v7;
      *v10 = v12;
      v10 = (__int64 *)v12;
      *a5 = v12;
LABEL_11:
      v7 = (__int64 *)v7[14];
      if ( !v7 )
        return v21;
    }
  }
  return result;
}
