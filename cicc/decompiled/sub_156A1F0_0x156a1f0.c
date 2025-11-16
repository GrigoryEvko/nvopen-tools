// Function: sub_156A1F0
// Address: 0x156a1f0
//
__int64 __fastcall sub_156A1F0(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r13
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 *v6; // r13
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 *v9; // r15
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  _BYTE *v14; // r12
  __int64 v15; // rbx
  _QWORD *v16; // rax
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rdx
  _QWORD *v21; // r8
  __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // rdx
  unsigned int v28; // r11d
  __int64 v29; // rsi
  unsigned int v30; // r12d
  _BYTE **v31; // rax
  int v32; // ebx
  __int64 v33; // r8
  _QWORD *v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rdx
  _QWORD *v37; // rdi
  __int64 result; // rax
  char v39; // r8
  __int64 v40; // [rsp+8h] [rbp-148h]
  _QWORD *v41; // [rsp+10h] [rbp-140h]
  __int64 v42; // [rsp+10h] [rbp-140h]
  _BYTE **v43; // [rsp+10h] [rbp-140h]
  __int64 v44; // [rsp+10h] [rbp-140h]
  __int64 v45; // [rsp+28h] [rbp-128h]
  _QWORD v46[2]; // [rsp+30h] [rbp-120h] BYREF
  _QWORD v47[2]; // [rsp+40h] [rbp-110h] BYREF
  __int16 v48; // [rsp+50h] [rbp-100h]
  _QWORD v49[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v50; // [rsp+70h] [rbp-E0h] BYREF
  _BYTE *v51; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+88h] [rbp-C8h]
  _BYTE v53[64]; // [rsp+90h] [rbp-C0h] BYREF
  _BYTE *v54; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v55; // [rsp+D8h] [rbp-78h]
  _BYTE v56[112]; // [rsp+E0h] [rbp-70h] BYREF

  v1 = a1;
  if ( *(_BYTE *)a1 != 4 )
    return a1;
  v2 = 8LL * *(unsigned int *)(a1 + 8);
  v3 = (__int64 *)(a1 - v2);
  v4 = v2 >> 3;
  v5 = v2 >> 5;
  if ( !v5 )
    goto LABEL_50;
  v6 = &v3[4 * v5];
  do
  {
    if ( (unsigned __int8)sub_15640D0(*v3) )
      goto LABEL_9;
    if ( (unsigned __int8)sub_15640D0(v3[1]) )
    {
      ++v3;
      goto LABEL_9;
    }
    if ( (unsigned __int8)sub_15640D0(v3[2]) )
    {
      v3 += 2;
      goto LABEL_9;
    }
    if ( (unsigned __int8)sub_15640D0(v3[3]) )
    {
      v3 += 3;
      goto LABEL_9;
    }
    v3 += 4;
  }
  while ( v6 != v3 );
  v4 = (a1 - (__int64)v3) >> 3;
LABEL_50:
  if ( v4 != 2 )
  {
    if ( v4 != 3 )
    {
      if ( v4 != 1 )
        return a1;
LABEL_62:
      v39 = sub_15640D0(*v3);
      result = a1;
      if ( !v39 )
        return result;
      goto LABEL_9;
    }
    if ( (unsigned __int8)sub_15640D0(*v3) )
      goto LABEL_9;
    ++v3;
  }
  if ( !(unsigned __int8)sub_15640D0(*v3) )
  {
    ++v3;
    goto LABEL_62;
  }
LABEL_9:
  if ( (__int64 *)a1 == v3 )
    return a1;
  v7 = *(unsigned int *)(a1 + 8);
  v51 = v53;
  v52 = 0x800000000LL;
  if ( v7 > 8 )
  {
    sub_16CD150(&v51, v53, v7, 8);
    v7 = *(unsigned int *)(a1 + 8);
  }
  v8 = 8 * v7;
  if ( a1 != a1 - v8 )
  {
    v9 = (__int64 *)(a1 - v8);
    while ( 1 )
    {
      v11 = *v9;
      if ( !*v9 )
        break;
      if ( *(_BYTE *)v11 == 4 )
      {
        v12 = *(unsigned int *)(v11 + 8);
        if ( (_DWORD)v12 )
        {
          v13 = -v12;
          v14 = *(_BYTE **)(v11 + 8 * v13);
          if ( v14 )
          {
            if ( !*v14 )
            {
              v15 = *v9;
              v16 = (_QWORD *)sub_161E970(*(_QWORD *)(v11 + 8 * v13));
              if ( v17 > 0xF && !(*v16 ^ 0x6365762E6D766C6CLL | v16[1] ^ 0x2E72657A69726F74LL) )
              {
                v54 = v56;
                v55 = 0x800000000LL;
                v18 = *(unsigned int *)(v11 + 8);
                if ( v18 > 8 )
                  sub_16CD150(&v54, v56, v18, 8);
                v19 = sub_161E970(v14);
                v21 = (_QWORD *)(*(_QWORD *)(v11 + 16) & 0xFFFFFFFFFFFFFFF8LL);
                if ( (*(_QWORD *)(v11 + 16) & 4) != 0 )
                  v21 = (_QWORD *)*v21;
                v22 = v20;
                if ( v20 == 22 )
                {
                  if ( *(_QWORD *)v19 ^ 0x6365762E6D766C6CLL | *(_QWORD *)(v19 + 8) ^ 0x2E72657A69726F74LL
                    || *(_DWORD *)(v19 + 16) != 1869770357
                    || *(_WORD *)(v19 + 20) != 27756 )
                  {
LABEL_27:
                    v23 = v20 - 16;
                    v22 = 16;
                    goto LABEL_28;
                  }
                  v25 = sub_161FF10(v21, "llvm.loop.interleave.count", 26);
                }
                else
                {
                  v23 = 0;
                  if ( v20 > 0xF )
                    goto LABEL_27;
LABEL_28:
                  v46[1] = v23;
                  v46[0] = v22 + v19;
                  v47[0] = "llvm.loop.vectorize.";
                  v47[1] = v46;
                  v41 = v21;
                  v48 = 1283;
                  sub_16E2FC0(v49, v47);
                  v24 = sub_161FF10(v41, v49[0], v49[1]);
                  v25 = v24;
                  if ( (__int64 *)v49[0] != &v50 )
                  {
                    v42 = v24;
                    j_j___libc_free_0(v49[0], v50 + 1);
                    v25 = v42;
                  }
                }
                v26 = (unsigned int)v55;
                if ( (unsigned int)v55 >= HIDWORD(v55) )
                {
                  v44 = v25;
                  sub_16CD150(&v54, v56, 0, 8);
                  v26 = (unsigned int)v55;
                  v25 = v44;
                }
                *(_QWORD *)&v54[8 * v26] = v25;
                v27 = (unsigned int)(v55 + 1);
                LODWORD(v55) = v55 + 1;
                v28 = *(_DWORD *)(v11 + 8);
                if ( v28 != 1 )
                {
                  v29 = v28;
                  v30 = 1;
                  v31 = &v54;
                  v32 = *(_DWORD *)(v11 + 8);
                  while ( 1 )
                  {
                    v33 = *(_QWORD *)(v11 + 8 * (v30 - v29));
                    if ( HIDWORD(v55) <= (unsigned int)v27 )
                    {
                      v40 = *(_QWORD *)(v11 + 8 * (v30 - v29));
                      v43 = v31;
                      sub_16CD150(v31, v56, 0, 8);
                      v27 = (unsigned int)v55;
                      v33 = v40;
                      v31 = v43;
                    }
                    ++v30;
                    *(_QWORD *)&v54[8 * v27] = v33;
                    v27 = (unsigned int)(v55 + 1);
                    LODWORD(v55) = v55 + 1;
                    if ( v32 == v30 )
                      break;
                    v29 = *(unsigned int *)(v11 + 8);
                  }
                }
                v34 = (_QWORD *)(*(_QWORD *)(v11 + 16) & 0xFFFFFFFFFFFFFFF8LL);
                if ( (*(_QWORD *)(v11 + 16) & 4) != 0 )
                  v34 = (_QWORD *)*v34;
                v15 = sub_1627350(v34, v54, v27, 0, 1);
                if ( v54 != v56 )
                  _libc_free((unsigned __int64)v54);
              }
LABEL_42:
              v35 = (unsigned int)v52;
              if ( (unsigned int)v52 >= HIDWORD(v52) )
                goto LABEL_56;
              goto LABEL_43;
            }
          }
        }
      }
      v35 = (unsigned int)v52;
      v15 = *v9;
      if ( (unsigned int)v52 >= HIDWORD(v52) )
      {
LABEL_56:
        sub_16CD150(&v51, v53, 0, 8);
        v35 = (unsigned int)v52;
      }
LABEL_43:
      ++v9;
      *(_QWORD *)&v51[8 * v35] = v15;
      v36 = (unsigned int)(v52 + 1);
      LODWORD(v52) = v52 + 1;
      if ( (__int64 *)a1 == v9 )
      {
        v1 = a1;
        goto LABEL_45;
      }
    }
    v15 = 0;
    goto LABEL_42;
  }
  v36 = (unsigned int)v52;
LABEL_45:
  v37 = (_QWORD *)(*(_QWORD *)(v1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(v1 + 16) & 4) != 0 )
    v37 = (_QWORD *)*v37;
  result = sub_1627350(v37, v51, v36, 0, 1);
  if ( v51 != v53 )
  {
    v45 = result;
    _libc_free((unsigned __int64)v51);
    return v45;
  }
  return result;
}
