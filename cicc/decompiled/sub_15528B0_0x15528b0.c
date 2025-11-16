// Function: sub_15528B0
// Address: 0x15528b0
//
__int64 __fastcall sub_15528B0(__int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  char v11; // si
  __int64 v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // rcx
  __int64 v15; // rdx
  __int64 **v16; // r15
  unsigned __int64 v17; // r12
  __int64 v18; // rdi
  _BYTE *v19; // rax
  __int64 v20; // rdi
  _BYTE *v21; // rax
  __int64 v22; // rdi
  _BYTE *v23; // rax
  __int64 **v24; // r12
  __int64 v25; // rdi
  _BYTE *v26; // rax
  __int64 *v27; // rsi
  __int64 v28; // rdi
  _WORD *v29; // rdx
  __int64 v30; // rdi
  _BYTE *v31; // rax
  __int64 v32; // rdi
  _WORD *v33; // rdx
  _QWORD *v34; // [rsp+0h] [rbp-50h]
  _QWORD *v35; // [rsp+0h] [rbp-50h]
  __int64 v36; // [rsp+8h] [rbp-48h]
  __int64 v37; // [rsp+18h] [rbp-38h]

  v3 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  result = *(unsigned __int8 *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (result & 0x80u) != 0LL )
  {
    result = sub_1648A40(v3);
    v6 = result + v5;
    if ( *(char *)(v3 + 23) < 0 )
    {
      result = sub_1648A40(v3);
      v6 -= result;
    }
    if ( (unsigned int)(v6 >> 4) )
    {
      sub_1263B40(*a1, " [ ");
      if ( *(char *)(v3 + 23) < 0 )
      {
        v7 = sub_1648A40(v3);
        v9 = v7 + v8;
        if ( *(char *)(v3 + 23) >= 0 )
          v10 = v9 >> 4;
        else
          LODWORD(v10) = (v9 - sub_1648A40(v3)) >> 4;
        if ( (_DWORD)v10 )
        {
          v37 = 0;
          v11 = 1;
          v36 = 16LL * (unsigned int)v10;
          do
          {
            v12 = 0;
            if ( *(char *)(v3 + 23) < 0 )
              v12 = sub_1648A40(v3);
            v13 = v37 + v12;
            v14 = *(_QWORD **)v13;
            v15 = 24LL * *(unsigned int *)(v13 + 8);
            v16 = (__int64 **)(v3 + v15 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF));
            v17 = 0xAAAAAAAAAAAAAAABLL * ((24LL * *(unsigned int *)(v13 + 12) - v15) >> 3);
            if ( !v11 )
            {
              v32 = *a1;
              v33 = *(_WORD **)(*a1 + 24);
              if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v33 <= 1u )
              {
                v35 = *(_QWORD **)v13;
                sub_16E7EE0(v32, ", ", 2);
                v14 = v35;
              }
              else
              {
                *v33 = 8236;
                *(_QWORD *)(v32 + 24) += 2LL;
              }
            }
            v18 = *a1;
            v19 = *(_BYTE **)(*a1 + 24);
            if ( (unsigned __int64)v19 >= *(_QWORD *)(*a1 + 16) )
            {
              v34 = v14;
              sub_16E7DE0(v18, 34);
              v14 = v34;
            }
            else
            {
              *(_QWORD *)(v18 + 24) = v19 + 1;
              *v19 = 34;
            }
            sub_16D16F0(v14 + 2, *v14, *a1);
            v20 = *a1;
            v21 = *(_BYTE **)(*a1 + 24);
            if ( (unsigned __int64)v21 >= *(_QWORD *)(*a1 + 16) )
            {
              sub_16E7DE0(v20, 34);
            }
            else
            {
              *(_QWORD *)(v20 + 24) = v21 + 1;
              *v21 = 34;
            }
            v22 = *a1;
            v23 = *(_BYTE **)(*a1 + 24);
            if ( (unsigned __int64)v23 >= *(_QWORD *)(*a1 + 16) )
            {
              sub_16E7DE0(v22, 40);
            }
            else
            {
              *(_QWORD *)(v22 + 24) = v23 + 1;
              *v23 = 40;
            }
            v24 = &v16[3 * v17];
            if ( v16 != v24 )
            {
              while ( 1 )
              {
                sub_154DAA0((__int64)(a1 + 5), **v16, *a1);
                v25 = *a1;
                v26 = *(_BYTE **)(*a1 + 24);
                if ( *(_BYTE **)(*a1 + 16) == v26 )
                {
                  sub_16E7EE0(v25, " ", 1);
                }
                else
                {
                  *v26 = 32;
                  ++*(_QWORD *)(v25 + 24);
                }
                v27 = *v16;
                v16 += 3;
                sub_1550E20(*a1, (__int64)v27, (__int64)(a1 + 5), a1[4], a1[1]);
                if ( v24 == v16 )
                  break;
                v28 = *a1;
                v29 = *(_WORD **)(*a1 + 24);
                if ( *(_QWORD *)(*a1 + 16) - (_QWORD)v29 <= 1u )
                {
                  sub_16E7EE0(v28, ", ", 2);
                }
                else
                {
                  *v29 = 8236;
                  *(_QWORD *)(v28 + 24) += 2LL;
                }
              }
            }
            v30 = *a1;
            v31 = *(_BYTE **)(*a1 + 24);
            if ( (unsigned __int64)v31 >= *(_QWORD *)(*a1 + 16) )
            {
              sub_16E7DE0(v30, 41);
            }
            else
            {
              *(_QWORD *)(v30 + 24) = v31 + 1;
              *v31 = 41;
            }
            v37 += 16;
            v11 = 0;
          }
          while ( v37 != v36 );
        }
      }
      return sub_1263B40(*a1, " ]");
    }
  }
  return result;
}
