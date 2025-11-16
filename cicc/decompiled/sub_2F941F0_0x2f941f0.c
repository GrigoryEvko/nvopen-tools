// Function: sub_2F941F0
// Address: 0x2f941f0
//
__int64 __fastcall sub_2F941F0(__int64 a1, unsigned __int64 a2, unsigned int a3)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r11
  __int64 v7; // r10
  __int64 v8; // rdx
  unsigned int v9; // edi
  unsigned int v10; // ebx
  __int64 v11; // rdx
  __int64 v12; // rcx
  _DWORD *v13; // rax
  __int64 v14; // rax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // r15
  __int64 v18; // r14
  __int64 result; // rax
  unsigned int v20; // r9d
  __int64 v21; // r12
  unsigned int v22; // ecx
  __int64 v23; // rsi
  __int64 v24; // rax
  _DWORD *v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // r13
  __int64 v28; // rbx
  _QWORD *v29; // r14
  __int64 v30; // rcx
  unsigned __int64 v31; // r8
  unsigned __int64 v32; // r9
  __int64 v33; // rcx
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r15
  __int64 v38; // r14
  int v39; // r8d
  __int64 v40; // rsi
  __int64 v41; // rdi
  __int64 v42; // r8
  __int64 v43; // rcx
  __int32 v44; // eax
  __int64 v45; // rcx
  unsigned __int64 v46; // r8
  __int64 v47; // r11
  __int64 v48; // r10
  void (*v49)(); // rax
  __int64 v50; // rdx
  __int64 v51; // r14
  __int64 v52; // r12
  __int64 v53; // r15
  unsigned int v54; // r14d
  unsigned int v55; // edi
  _DWORD *v56; // rcx
  __int64 v57; // r8
  __int64 v58; // rdi
  __int64 v59; // rdx
  __int64 v60; // [rsp+0h] [rbp-B0h]
  __int64 v61; // [rsp+8h] [rbp-A8h]
  __int64 v62; // [rsp+10h] [rbp-A0h]
  __int64 v63; // [rsp+10h] [rbp-A0h]
  __int64 v64; // [rsp+10h] [rbp-A0h]
  __int64 v65; // [rsp+18h] [rbp-98h]
  __int128 v66; // [rsp+20h] [rbp-90h]
  __int64 v67; // [rsp+30h] [rbp-80h]
  __int64 v68; // [rsp+38h] [rbp-78h]
  __int64 v69; // [rsp+38h] [rbp-78h]
  unsigned int v71; // [rsp+44h] [rbp-6Ch]
  unsigned __int64 v73; // [rsp+50h] [rbp-60h] BYREF
  unsigned int v74; // [rsp+58h] [rbp-58h]
  int v75; // [rsp+5Ch] [rbp-54h]
  __m128i v76; // [rsp+60h] [rbp-50h] BYREF
  unsigned __int64 v77; // [rsp+70h] [rbp-40h]
  _QWORD *v78; // [rsp+78h] [rbp-38h]

  v67 = *(_QWORD *)a2;
  v4 = *(_QWORD *)(*(_QWORD *)a2 + 32LL) + 40LL * a3;
  v71 = *(_DWORD *)(v4 + 8);
  if ( *(_BYTE *)(a1 + 899) )
  {
    if ( (*(_DWORD *)v4 & 0xFFF00) == 0 || (*(_BYTE *)(v4 + 4) & 1) != 0 )
    {
      v5 = sub_2F91CB0(a1, (_DWORD *)v4);
      v6 = -1;
      v7 = -1;
      *((_QWORD *)&v66 + 1) = v5;
      *(_QWORD *)&v66 = v8;
    }
    else
    {
      v35 = sub_2F91CB0(a1, (_DWORD *)v4);
      *(_QWORD *)&v66 = v36;
      v7 = v35;
      v6 = v36;
      *((_QWORD *)&v66 + 1) = v35;
    }
    if ( (*(_DWORD *)v4 & 0xFFF00) != 0 && (*(_BYTE *)(v4 + 4) & 1) != 0 )
    {
      v50 = *(_QWORD *)(v67 + 32);
      v51 = v50 + 40LL * (*(_DWORD *)(v67 + 40) & 0xFFFFFF);
      v52 = v50 + 40LL * (a3 + 1);
      if ( v51 != v52 )
      {
        v53 = v6;
        do
        {
          if ( !*(_BYTE *)v52 && (*(_BYTE *)(v52 + 3) & 0x10) != 0 && v71 == *(_DWORD *)(v52 + 8) )
          {
            v7 &= ~sub_2F91CB0(a1, (_DWORD *)v52);
            v53 &= ~v59;
          }
          v52 += 40;
        }
        while ( v51 != v52 );
        v6 = v53;
      }
    }
    *(_BYTE *)(v4 + 4) &= ~1u;
  }
  else
  {
    v6 = -1;
    v7 = -1;
    *(_QWORD *)&v66 = -1;
    *((_QWORD *)&v66 + 1) = -1;
  }
  if ( (((*(_BYTE *)(v4 + 3) & 0x10) != 0) & (*(_BYTE *)(v4 + 3) >> 6)) == 0 )
  {
    v9 = *(_DWORD *)(a1 + 1800);
    v60 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL);
    v10 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 2192) + (v71 & 0x7FFFFFFF));
    if ( v10 < v9 )
    {
      v11 = *(_QWORD *)(a1 + 1792);
      while ( 1 )
      {
        v12 = v10;
        v13 = (_DWORD *)(v11 + 48LL * v10);
        if ( (v71 & 0x7FFFFFFF) == (*v13 & 0x7FFFFFFF) )
        {
          v14 = (unsigned int)v13[10];
          if ( (_DWORD)v14 != -1 && *(_DWORD *)(v11 + 48 * v14 + 44) == -1 )
            break;
        }
        v10 += 256;
        if ( v9 <= v10 )
          goto LABEL_21;
      }
      if ( v10 != -1 )
      {
        while ( 1 )
        {
          v15 = 48 * v12;
          v16 = v11 + 48 * v12;
          v17 = *(_QWORD *)(v16 + 8);
          v18 = *(_QWORD *)(v16 + 16);
          if ( v18 & v6 | v17 & v7 )
          {
            if ( v18 & (unsigned __int64)v66 | v17 & *((_QWORD *)&v66 + 1) )
            {
              v61 = v7;
              v63 = v6;
              v43 = **(_QWORD **)(v16 + 24);
              v69 = *(_QWORD *)(v16 + 24);
              v76.m128i_i64[0] = a2 & 0xFFFFFFFFFFFFFFF9LL;
              v76.m128i_i64[1] = v71 | 0x100000000LL;
              v44 = sub_2FF8170(a1 + 600, v67, a3, v43, *(unsigned int *)(v16 + 32));
              v47 = v63;
              v76.m128i_i32[3] = v44;
              v48 = v61;
              v49 = *(void (**)())(*(_QWORD *)v60 + 344LL);
              if ( v49 != nullsub_1667 )
              {
                ((void (__fastcall *)(__int64, unsigned __int64, _QWORD, __int64, _QWORD))v49)(
                  v60,
                  a2,
                  a3,
                  v69,
                  *(unsigned int *)(*(_QWORD *)(a1 + 1792) + v15 + 32));
                v48 = v61;
                v47 = v63;
              }
              v64 = v48;
              v65 = v47;
              sub_2F8F1B0(v69, (__int64)&v76, 1u, v45, v46, (unsigned __int64)&v76);
              v11 = *(_QWORD *)(a1 + 1792);
              v7 = v64;
              v6 = v65;
              v16 = v11 + v15;
            }
            v37 = ~v7 & v17;
            v38 = ~v6 & v18;
            if ( v38 | v37 )
            {
              *(_QWORD *)(v16 + 8) = v37;
              *(_QWORD *)(v16 + 16) = v38;
              v10 = *(_DWORD *)(*(_QWORD *)(a1 + 1792) + v15 + 44);
            }
            else
            {
              v39 = -1;
              v40 = *(unsigned int *)(v16 + 40);
              v41 = v11 + 48 * v40;
              if ( v16 != v41 )
              {
                v42 = *(unsigned int *)(v16 + 44);
                if ( *(_DWORD *)(v41 + 44) == -1 )
                {
                  *(_BYTE *)(*(_QWORD *)(a1 + 2192) + (*(_DWORD *)v16 & 0x7FFFFFFF)) = v42;
                  *(_DWORD *)(*(_QWORD *)(a1 + 1792) + 48LL * *(unsigned int *)(v16 + 44) + 40) = *(_DWORD *)(v16 + 40);
                  v39 = *(_DWORD *)(v16 + 44);
                  v41 = v15 + *(_QWORD *)(a1 + 1792);
                }
                else if ( (_DWORD)v42 == -1 )
                {
                  v54 = *(_DWORD *)(a1 + 1800);
                  v55 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 2192) + (*(_DWORD *)v16 & 0x7FFFFFFF));
                  if ( v55 < v54 )
                  {
                    while ( 1 )
                    {
                      v56 = (_DWORD *)(v11 + 48LL * v55);
                      if ( (*(_DWORD *)v16 & 0x7FFFFFFF) == (*v56 & 0x7FFFFFFF) )
                      {
                        v57 = (unsigned int)v56[10];
                        if ( (_DWORD)v57 != -1 && *(_DWORD *)(v11 + 48 * v57 + 44) == -1 )
                          break;
                      }
                      v55 += 256;
                      if ( v54 <= v55 )
                        goto LABEL_69;
                    }
                  }
                  else
                  {
LABEL_69:
                    v56 = (_DWORD *)(v11 + 0x2FFFFFFFD0LL);
                  }
                  v56[10] = v40;
                  *(_DWORD *)(*(_QWORD *)(a1 + 1792) + 48LL * *(unsigned int *)(v16 + 40) + 44) = *(_DWORD *)(v16 + 44);
                  v58 = *(_QWORD *)(a1 + 1792);
                  v39 = *(_DWORD *)(v58 + 48LL * *(unsigned int *)(v16 + 40) + 44);
                  v41 = v15 + v58;
                }
                else
                {
                  *(_DWORD *)(v11 + 48 * v42 + 40) = v40;
                  v39 = *(_DWORD *)(v16 + 44);
                  *(_DWORD *)(*(_QWORD *)(a1 + 1792) + 48 * v40 + 44) = v39;
                  v41 = v15 + *(_QWORD *)(a1 + 1792);
                }
              }
              *(_DWORD *)(v41 + 40) = -1;
              *(_DWORD *)(*(_QWORD *)(a1 + 1792) + v15 + 44) = *(_DWORD *)(a1 + 2208);
              *(_DWORD *)(a1 + 2208) = v10;
              v10 = v39;
              ++*(_DWORD *)(a1 + 2212);
            }
          }
          else
          {
            v10 = *(_DWORD *)(v16 + 44);
          }
          if ( v10 == -1 )
            break;
          v11 = *(_QWORD *)(a1 + 1792);
          v12 = v10;
        }
      }
    }
  }
LABEL_21:
  result = sub_2DADE10(*(_QWORD *)(a1 + 40), v71);
  if ( !(_BYTE)result )
  {
    v20 = *(_DWORD *)(a1 + 1440);
    v21 = a1 + 1432;
    v22 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 1768) + (v71 & 0x7FFFFFFF));
    if ( v22 < v20 )
    {
      v23 = *(_QWORD *)(a1 + 1432);
      while ( 1 )
      {
        v24 = v22;
        v25 = (_DWORD *)(v23 + 40LL * v22);
        if ( (v71 & 0x7FFFFFFF) == (*v25 & 0x7FFFFFFF) )
        {
          v26 = (unsigned int)v25[8];
          if ( (_DWORD)v26 != -1 && *(_DWORD *)(v23 + 40 * v26 + 36) == -1 )
            break;
        }
        v22 += 256;
        if ( v20 <= v22 )
          goto LABEL_33;
      }
      if ( v22 != -1 )
      {
        v68 = a1;
        v62 = a1 + 1432;
        do
        {
          v27 = 40 * v24;
          v28 = v23 + 40 * v24;
          if ( (unsigned __int64)v66 & *(_QWORD *)(v28 + 16) | *((_QWORD *)&v66 + 1) & *(_QWORD *)(v28 + 8) )
          {
            v29 = *(_QWORD **)(v28 + 24);
            if ( (_QWORD *)a2 != v29 )
            {
              v75 = 0;
              v73 = a2 & 0xFFFFFFFFFFFFFFF9LL | 4;
              v74 = v71;
              v75 = sub_2FF8480(v68 + 600, v67, a3, *v29);
              sub_2F8F1B0((__int64)v29, (__int64)&v73, 1u, v30, v31, v32);
              v33 = *(_QWORD *)(v28 + 8) & ~*((_QWORD *)&v66 + 1);
              v34 = *(_QWORD *)(v28 + 16) & ~(_QWORD)v66;
              *(_QWORD *)(v28 + 24) = a2;
              *(_QWORD *)(v28 + 8) &= *((_QWORD *)&v66 + 1);
              *(_QWORD *)(v28 + 16) &= v66;
              if ( __PAIR128__(v33, v34) != 0 )
              {
                v76.m128i_i64[1] = v33;
                v77 = v34;
                v76.m128i_i32[0] = v71;
                v78 = v29;
                sub_2ECA8A0(v62, &v76);
              }
              v23 = *(_QWORD *)(v68 + 1432);
              v28 = v23 + v27;
            }
          }
          v24 = *(unsigned int *)(v28 + 36);
        }
        while ( (_DWORD)v24 != -1 );
        v21 = v62;
      }
    }
LABEL_33:
    result = v66 | *((_QWORD *)&v66 + 1);
    if ( v66 != 0 )
    {
      v76.m128i_i32[0] = v71;
      v76.m128i_i64[1] = *((_QWORD *)&v66 + 1);
      v77 = v66;
      v78 = (_QWORD *)a2;
      return sub_2ECA8A0(v21, &v76);
    }
  }
  return result;
}
