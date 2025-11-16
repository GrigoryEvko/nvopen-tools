// Function: sub_19129F0
// Address: 0x19129f0
//
__int64 __fastcall sub_19129F0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  __int64 v5; // rcx
  int v6; // r8d
  int v7; // r9d
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  unsigned int v11; // esi
  __int64 v12; // r8
  unsigned int v13; // edx
  __int64 *v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // r13d
  unsigned __int64 v18; // rax
  unsigned int v19; // ebx
  __int64 v20; // rdi
  unsigned __int64 v21; // rax
  int v22; // edx
  __int64 *v23; // rax
  unsigned __int64 v24; // rdi
  int v25; // edi
  int v26; // r11d
  __int64 *v27; // r10
  int v28; // ecx
  const __m128i **v29; // r14
  const __m128i *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  unsigned __int64 v33; // r13
  __int64 v34; // rbx
  __int64 *v35; // rax
  unsigned __int64 v36; // rdx
  bool v37; // al
  int v38; // r14d
  __int64 v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rbx
  int v42; // ebx
  __int64 v43; // rax
  __int64 v44; // rdx
  int v45; // ebx
  __int64 v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rdx
  int v50; // ebx
  __int64 v51; // r14
  unsigned int v52; // eax
  int v53; // ebx
  unsigned __int64 v54; // rax
  int v55; // r14d
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rbx
  int v59; // ebx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rbx
  int v65; // ebx
  __int64 v66; // rax
  __int64 v67; // rdx
  int v68; // ebx
  __int64 v69; // r14
  unsigned int v70; // eax
  int v71; // ebx
  __int64 v72; // [rsp+8h] [rbp-88h]
  int v73; // [rsp+8h] [rbp-88h]
  __int64 v74; // [rsp+8h] [rbp-88h]
  int v75; // [rsp+8h] [rbp-88h]
  __int64 v76; // [rsp+8h] [rbp-88h]
  __int64 v77; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v78[3]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v79; // [rsp+38h] [rbp-58h]
  _BYTE v80[72]; // [rsp+48h] [rbp-48h] BYREF

  v2 = a2;
  v3 = a2 | 4;
  if ( (unsigned int)sub_134CC90(*(_QWORD *)(a1 + 184), a2 | 4) == 4 )
  {
    sub_19127E0((__int64)v78, a1, a2, v5, v6, v7);
    v77 = a2;
    v16 = sub_1911DB0(a1, (__int64)v78);
    *((_DWORD *)sub_1910D10(a1, &v77) + 2) = v16;
    v24 = (unsigned __int64)v79;
    if ( v79 == v80 )
      return v16;
    goto LABEL_14;
  }
  if ( *(_QWORD *)(a1 + 192) && (sub_134CC90(*(_QWORD *)(a1 + 184), v3) & 2) == 0 )
  {
    sub_19127E0((__int64)v78, a1, a2, v8, v9, v10);
    v18 = sub_1911DB0(a1, (__int64)v78);
    v19 = v18;
    if ( !BYTE4(v18) )
    {
      v20 = *(_QWORD *)(a1 + 192);
      if ( v20 )
      {
        v21 = sub_141C430(v20, a2, 1u);
        v22 = v21 & 7;
        if ( v22 != 2 )
        {
          if ( v22 != 3 )
            goto LABEL_12;
          if ( v21 >> 61 != 1 )
            goto LABEL_12;
          v29 = sub_1418110(*(_QWORD *)(a1 + 192), v3);
          v30 = *v29;
          v31 = v29[1] - *v29;
          if ( !(_DWORD)v31 )
            goto LABEL_12;
          v32 = 0;
          v33 = 0;
          v34 = 16LL * (unsigned int)(v31 - 1);
          while ( 1 )
          {
            v35 = (__int64 *)((char *)v30->m128i_i64 + v32);
            v36 = v35[1];
            if ( (v36 & 7) == 3 )
            {
              if ( v36 >> 61 != 1 )
                goto LABEL_12;
            }
            else
            {
              v72 = v32;
              if ( v33 )
                goto LABEL_12;
              if ( (v36 & 7) != 2 )
                goto LABEL_12;
              v33 = v36 & 0xFFFFFFFFFFFFFFF8LL;
              if ( *(_BYTE *)((v36 & 0xFFFFFFFFFFFFFFF8LL) + 16) != 78 )
                goto LABEL_12;
              v37 = sub_15CC890(*(_QWORD *)(a1 + 200), *v35, *(_QWORD *)(a2 + 40));
              v32 = v72;
              if ( !v37 )
                goto LABEL_12;
            }
            if ( v34 == v32 )
              break;
            v30 = *v29;
            v32 += 16;
          }
          if ( !v33 )
            goto LABEL_12;
          v38 = *(_DWORD *)(v33 + 20) & 0xFFFFFFF;
          if ( *(char *)(v33 + 23) < 0 )
          {
            v39 = sub_1648A40(v33);
            v41 = v39 + v40;
            if ( *(char *)(v33 + 23) >= 0 )
            {
              if ( !(unsigned int)(v41 >> 4) )
                goto LABEL_49;
            }
            else
            {
              if ( !(unsigned int)((v41 - sub_1648A40(v33)) >> 4) )
                goto LABEL_49;
              if ( *(char *)(v33 + 23) < 0 )
              {
                v42 = *(_DWORD *)(sub_1648A40(v33) + 8);
                if ( *(char *)(v33 + 23) >= 0 )
                  BUG();
                v43 = sub_1648A40(v33);
                v38 += v42 - *(_DWORD *)(v43 + v44 - 4);
                goto LABEL_49;
              }
            }
            BUG();
          }
LABEL_49:
          v45 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
          if ( *(char *)(a2 + 23) >= 0 )
            goto LABEL_55;
          v46 = sub_1648A40(a2);
          if ( *(char *)(a2 + 23) >= 0 )
          {
            if ( !(unsigned int)((v46 + v47) >> 4) )
              goto LABEL_55;
          }
          else
          {
            if ( !(unsigned int)((v46 + v47 - sub_1648A40(a2)) >> 4) )
              goto LABEL_55;
            if ( *(char *)(a2 + 23) < 0 )
            {
              v73 = *(_DWORD *)(sub_1648A40(a2) + 8);
              if ( *(char *)(a2 + 23) >= 0 )
                BUG();
              v48 = sub_1648A40(a2);
              v45 += v73 - *(_DWORD *)(v48 + v49 - 4);
LABEL_55:
              if ( v45 == v38 )
              {
                v50 = *(_DWORD *)(a2 + 20);
                v51 = 0;
                v52 = (v50 & 0xFFFFFFF) - 1 - sub_154CB40(a2);
                v74 = v52;
                if ( v52 )
                {
                  while ( 1 )
                  {
                    v53 = sub_1911FD0(a1, *(_QWORD *)(a2 + 24 * (v51 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
                    if ( v53 != (unsigned int)sub_1911FD0(
                                                a1,
                                                *(_QWORD *)(v33 + 24 * (v51 - (*(_DWORD *)(v33 + 20) & 0xFFFFFFF)))) )
                      goto LABEL_12;
                    if ( v74 == ++v51 )
                      goto LABEL_59;
                  }
                }
                goto LABEL_59;
              }
              goto LABEL_12;
            }
          }
          BUG();
        }
        v54 = v21 & 0xFFFFFFFFFFFFFFF8LL;
        v33 = v54;
        v55 = *(_DWORD *)(v54 + 20) & 0xFFFFFFF;
        if ( *(char *)(v54 + 23) < 0 )
        {
          v56 = sub_1648A40(v54);
          v58 = v56 + v57;
          if ( *(char *)(v33 + 23) >= 0 )
          {
            if ( !(unsigned int)(v58 >> 4) )
              goto LABEL_66;
          }
          else
          {
            if ( !(unsigned int)((v58 - sub_1648A40(v33)) >> 4) )
              goto LABEL_66;
            if ( *(char *)(v33 + 23) < 0 )
            {
              v59 = *(_DWORD *)(sub_1648A40(v33) + 8);
              if ( *(char *)(v33 + 23) >= 0 )
                BUG();
              v60 = sub_1648A40(v33);
              v55 += v59 - *(_DWORD *)(v60 + v61 - 4);
              goto LABEL_66;
            }
          }
          BUG();
        }
LABEL_66:
        v75 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        if ( *(char *)(a2 + 23) >= 0 )
          goto LABEL_72;
        v62 = sub_1648A40(a2);
        v64 = v62 + v63;
        if ( *(char *)(a2 + 23) >= 0 )
        {
          if ( !(unsigned int)(v64 >> 4) )
            goto LABEL_72;
        }
        else
        {
          if ( !(unsigned int)((v64 - sub_1648A40(a2)) >> 4) )
            goto LABEL_72;
          if ( *(char *)(a2 + 23) < 0 )
          {
            v65 = *(_DWORD *)(sub_1648A40(a2) + 8);
            if ( *(char *)(a2 + 23) >= 0 )
              BUG();
            v66 = sub_1648A40(a2);
            v75 += v65 - *(_DWORD *)(v66 + v67 - 4);
LABEL_72:
            if ( v55 == v75 )
            {
              v68 = *(_DWORD *)(a2 + 20);
              v69 = 0;
              v70 = (v68 & 0xFFFFFFF) - 1 - sub_154CB40(a2);
              v76 = v70;
              if ( v70 )
              {
                do
                {
                  v71 = sub_1911FD0(a1, *(_QWORD *)(a2 + 24 * (v69 - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
                  if ( v71 != (unsigned int)sub_1911FD0(
                                              a1,
                                              *(_QWORD *)(v33 + 24 * (v69 - (*(_DWORD *)(v33 + 20) & 0xFFFFFFF)))) )
                    goto LABEL_12;
                }
                while ( v76 != ++v69 );
              }
LABEL_59:
              v77 = a2;
              v16 = sub_1911FD0(a1, v33);
              *((_DWORD *)sub_1910D10(a1, &v77) + 2) = v16;
LABEL_13:
              v24 = (unsigned __int64)v79;
              if ( v79 == v80 )
                return v16;
LABEL_14:
              _libc_free(v24);
              return v16;
            }
LABEL_12:
            v77 = a2;
            v23 = sub_1910D10(a1, &v77);
            v16 = *(_DWORD *)(a1 + 208);
            *((_DWORD *)v23 + 2) = v16;
            *(_DWORD *)(a1 + 208) = v16 + 1;
            goto LABEL_13;
          }
        }
        BUG();
      }
      v19 = sub_1911DB0(a1, (__int64)v78);
    }
    v77 = a2;
    v16 = v19;
    *((_DWORD *)sub_1910D10(a1, &v77) + 2) = v19;
    goto LABEL_13;
  }
  v11 = *(_DWORD *)(a1 + 24);
  v77 = v2;
  if ( !v11 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_20;
  }
  v12 = *(_QWORD *)(a1 + 8);
  v13 = (v11 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( v2 != *v14 )
  {
    v26 = 1;
    v27 = 0;
    while ( v15 != -8 )
    {
      if ( !v27 && v15 == -16 )
        v27 = v14;
      v13 = (v11 - 1) & (v26 + v13);
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v2 == *v14 )
        goto LABEL_6;
      ++v26;
    }
    v28 = *(_DWORD *)(a1 + 16);
    if ( v27 )
      v14 = v27;
    ++*(_QWORD *)a1;
    v25 = v28 + 1;
    if ( 4 * (v28 + 1) < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(a1 + 20) - v25 > v11 >> 3 )
        goto LABEL_22;
      goto LABEL_21;
    }
LABEL_20:
    v11 *= 2;
LABEL_21:
    sub_177C7D0(a1, v11);
    sub_190E590(a1, &v77, v78);
    v14 = (__int64 *)v78[0];
    v2 = v77;
    v25 = *(_DWORD *)(a1 + 16) + 1;
LABEL_22:
    *(_DWORD *)(a1 + 16) = v25;
    if ( *v14 != -8 )
      --*(_DWORD *)(a1 + 20);
    *v14 = v2;
    *((_DWORD *)v14 + 2) = 0;
  }
LABEL_6:
  v16 = *(_DWORD *)(a1 + 208);
  *((_DWORD *)v14 + 2) = v16;
  *(_DWORD *)(a1 + 208) = v16 + 1;
  return v16;
}
