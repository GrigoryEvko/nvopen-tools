// Function: sub_6F1E90
// Address: 0x6f1e90
//
__int64 __fastcall sub_6F1E90(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  _QWORD *v7; // rbx
  __int64 *v8; // rax
  __int64 v9; // r14
  __int64 i; // r11
  __int64 v11; // r12
  bool v12; // al
  char v13; // al
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // r14
  __int64 v18; // r12
  __int64 j; // rax
  __int64 v21; // rdi
  __int64 v22; // r13
  __int64 v23; // rax
  __int64 *v24; // rax
  __int64 v25; // rax
  _DWORD *v26; // r13
  __int64 v27; // [rsp-8h] [rbp-88h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  int v29; // [rsp+10h] [rbp-70h]
  bool v30; // [rsp+15h] [rbp-6Bh]
  __int16 v31; // [rsp+16h] [rbp-6Ah]
  int v32; // [rsp+18h] [rbp-68h]
  _QWORD *v33; // [rsp+20h] [rbp-60h]
  unsigned int v34; // [rsp+34h] [rbp-4Ch] BYREF
  __int64 v35; // [rsp+38h] [rbp-48h] BYREF
  __m128i v36[4]; // [rsp+40h] [rbp-40h] BYREF

  v6 = a3;
  v7 = (_QWORD *)a2;
  v8 = *(__int64 **)a3;
  v34 = 0;
  v9 = *(_QWORD *)(a1 + 56);
  for ( i = *(_QWORD *)(a1 + 64); v8; v8 = (__int64 *)*v8 )
    ++*((_DWORD *)v8 + 6);
  v33 = *(_QWORD **)(a3 + 8);
  v29 = *(_DWORD *)(a3 + 72);
  if ( i && *(_QWORD *)(a2 + 16) && (*(_DWORD *)(a3 + 72) = 0, sub_8A4D00(i, a2, a1 + 28, 0, &v34, a3), (a5 = v34) != 0) )
  {
LABEL_22:
    LODWORD(v11) = 0;
  }
  else
  {
    LODWORD(v11) = 1;
    if ( v9 )
    {
      do
      {
        v13 = *(_BYTE *)(v9 + 24);
        switch ( v13 )
        {
          case 34:
            v21 = *(_QWORD *)(v9 + 56);
            a2 = (__int64)v7;
            v22 = *(_QWORD *)(v21 + 16);
            v23 = sub_6F02E0(v21, v7, v6, v22 != 0, (_BOOL4 *)v36[0].m128i_i32, a6);
            if ( v23 )
            {
              if ( (*(_BYTE *)(v9 + 64) & 1) == 0 || (a2 = v36[0].m128i_u32[0], v36[0].m128i_i32[0]) )
              {
                if ( !v22 )
                  goto LABEL_9;
                a2 = v22;
                if ( (unsigned int)sub_6F1D40(v23, v22, (int)v7, v6, 0) )
                  goto LABEL_9;
              }
            }
            v12 = 1;
            LODWORD(v11) = 0;
            break;
          case 35:
            v14 = v7[2];
            v15 = *(_QWORD *)(v9 + 56);
            v36[0] = 0u;
            if ( (_DWORD)v14 )
            {
              v32 = dword_4F07508[0];
              v31 = dword_4F07508[1];
              if ( (int)v14 > 1 )
              {
                v16 = qword_4F04C68[0] + 776LL * dword_4F04C64;
                v30 = (*(_BYTE *)(v16 + 6) & 4) != 0;
                if ( (*(_BYTE *)(v16 + 6) & 4) == 0 )
                  *(_BYTE *)(v16 + 6) |= 4u;
                v28 = v9;
                v17 = 0;
                v18 = 8 * (3LL * (unsigned int)(v14 - 2) + 3);
                do
                {
                  v15 = sub_743530(v15, *(_QWORD *)(v17 + *v7 + 8), *(_QWORD *)(v17 + *v7), 0x4000, &v34, v6);
                  if ( v34 )
                  {
                    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) = *(_BYTE *)(qword_4F04C68[0]
                                                                                        + 776LL * dword_4F04C64
                                                                                        + 6)
                                                                             & 0xFB
                                                                             | (4 * v30);
                    goto LABEL_22;
                  }
                  v17 += 24;
                }
                while ( v18 != v17 );
                v9 = v28;
                *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 6) = (4 * v30)
                                                                         | *(_BYTE *)(qword_4F04C68[0]
                                                                                    + 776LL * dword_4F04C64
                                                                                    + 6)
                                                                         & 0xFB;
                v14 = v7[2];
              }
              *(_QWORD *)dword_4F07508 = *(_QWORD *)(v9 + 28);
              v24 = (__int64 *)(*v7 + 8 * (3 * v14 - 3));
              a2 = v24[1];
              LODWORD(v11) = sub_6F1C10(v15, a2, *v24, v36, 0, (__int64 *)v6, 0, 0);
              sub_67E3D0(v36);
              dword_4F07508[0] = v32;
              LOWORD(dword_4F07508[1]) = v31;
              a3 = v27;
            }
            else
            {
              v25 = sub_724DC0(v15, a2, a3, a4, a5, a6);
              a2 = 1;
              v35 = v25;
              if ( (unsigned int)sub_7A30C0(v15, 1, 1, v25) )
              {
                v11 = (unsigned int)sub_711520(v35, 1) == 0;
              }
              else
              {
                a2 = (__int64)v36;
                v26 = sub_67D9D0(0x1Cu, (_DWORD *)(v15 + 28));
                sub_67E370((__int64)v26, v36);
                sub_685910((__int64)v26, (FILE *)v36);
              }
              sub_724E30(&v35);
            }
LABEL_9:
            v12 = (_DWORD)v11 == 0;
            break;
          case 22:
            if ( v7[2] )
            {
              a2 = (__int64)v7;
              sub_8A4E60(*(_QWORD *)(v9 + 56), v7, v9 + 28, 0, &v34, v6);
              if ( v34 )
                goto LABEL_22;
            }
            goto LABEL_9;
          default:
            if ( !v7[2] )
              goto LABEL_9;
            a2 = (__int64)v7;
            if ( sub_6F02E0(v9, v7, v6, 0, (_BOOL4 *)v36[0].m128i_i32, a6) )
              goto LABEL_9;
            v12 = 1;
            LODWORD(v11) = 0;
            break;
        }
        v9 = *(_QWORD *)(v9 + 16);
      }
      while ( v9 && !v12 );
    }
  }
  if ( *(_QWORD *)v6 )
  {
    if ( v33 )
    {
      sub_8921C0(*v33);
      *v33 = 0;
      for ( j = *(_QWORD *)v6; j; j = *(_QWORD *)j )
        --*(_DWORD *)(j + 24);
      *(_QWORD *)(v6 + 8) = v33;
    }
    else
    {
      ((void (*)(void))sub_8921C0)();
      *(_QWORD *)v6 = 0;
      *(_QWORD *)(v6 + 8) = 0;
    }
  }
  *(_DWORD *)(v6 + 72) = v29;
  return (unsigned int)v11;
}
