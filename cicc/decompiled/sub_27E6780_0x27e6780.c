// Function: sub_27E6780
// Address: 0x27e6780
//
__int64 __fastcall sub_27E6780(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rbx
  _QWORD *v4; // r12
  __int64 v5; // rax
  _QWORD *v6; // r15
  __int64 v7; // r13
  unsigned __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // r14d
  unsigned __int64 v13; // rax
  __int64 v14; // r13
  int v15; // r14d
  unsigned int v16; // ebx
  unsigned int v17; // r12d
  unsigned int v18; // r14d
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rbx
  unsigned __int64 v26; // rax
  __int64 v27; // rcx
  _DWORD *v28; // rax
  _DWORD *v29; // r14
  unsigned int v30; // esi
  __int64 v31; // rdi
  int v32; // eax
  bool v33; // al
  _BOOL8 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // rdx
  __int64 v38; // rcx
  unsigned __int8 *v39; // rax
  unsigned int v40; // r13d
  unsigned __int8 *v41; // rax
  unsigned int v42; // eax
  unsigned int v43; // edx
  unsigned int v44; // ecx
  int v45; // eax
  bool v46; // al
  unsigned int v47; // [rsp+Ch] [rbp-84h]
  char *v48; // [rsp+10h] [rbp-80h]
  __int64 v49; // [rsp+18h] [rbp-78h]
  __int64 v50; // [rsp+20h] [rbp-70h]
  __int64 v51; // [rsp+28h] [rbp-68h]
  int v52; // [rsp+30h] [rbp-60h]
  unsigned __int64 v53; // [rsp+38h] [rbp-58h]
  int v54; // [rsp+38h] [rbp-58h]
  int v55; // [rsp+40h] [rbp-50h]
  unsigned __int64 v56; // [rsp+40h] [rbp-50h]
  __int64 v57; // [rsp+48h] [rbp-48h]

  v3 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v3 == a2 + 48 || !v3 || (unsigned int)*(unsigned __int8 *)(v3 - 24) - 30 > 0xA )
    goto LABEL_81;
  if ( *(_BYTE *)(v3 - 24) != 31 )
    return 0;
  v4 = (_QWORD *)a2;
  v5 = sub_AA54C0(a2);
  v6 = (_QWORD *)v5;
  if ( !v5 )
    return 0;
  v7 = v5 + 48;
  v8 = *(_QWORD *)(v5 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 == v8 || !v8 || (unsigned int)*(unsigned __int8 *)(v8 - 24) - 30 > 0xA )
    goto LABEL_81;
  if ( *(_BYTE *)(v8 - 24) != 31 )
    return 0;
  if ( (*(_DWORD *)(v8 - 20) & 0x7FFFFFF) == 1 )
    return 0;
  v57 = sub_AA54C0((__int64)v6);
  if ( v57 )
    return 0;
  v13 = v6[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 != v13 )
  {
    if ( !v13 )
      BUG();
    v14 = v13 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 <= 0xA )
    {
      v55 = sub_B46E30(v14);
      v15 = v55 >> 2;
      if ( v55 >> 2 > 0 )
      {
        v53 = v3;
        v16 = 0;
        while ( 1 )
        {
          if ( v6 == (_QWORD *)sub_B46EC0(v14, v16) )
          {
            v18 = v16;
            v4 = (_QWORD *)a2;
            v3 = v53;
            goto LABEL_25;
          }
          if ( v6 == (_QWORD *)sub_B46EC0(v14, v16 + 1) )
            break;
          v17 = v16 + 2;
          if ( v6 == (_QWORD *)sub_B46EC0(v14, v16 + 2) || (v17 = v16 + 3, v6 == (_QWORD *)sub_B46EC0(v14, v16 + 3)) )
          {
            v18 = v17;
            v3 = v53;
            v4 = (_QWORD *)a2;
            goto LABEL_25;
          }
          v16 += 4;
          if ( !--v15 )
          {
            v10 = v16;
            v18 = v16;
            v4 = (_QWORD *)a2;
            v3 = v53;
            v45 = v55 - v10;
            goto LABEL_62;
          }
        }
        v18 = v16 + 1;
        v3 = v53;
        v4 = (_QWORD *)a2;
LABEL_25:
        v10 = (unsigned int)v55;
        if ( v18 != v55 )
          return 0;
        goto LABEL_26;
      }
      v45 = v55;
      v18 = 0;
LABEL_62:
      if ( v45 != 2 )
      {
        if ( v45 != 3 )
        {
          if ( v45 != 1 )
            goto LABEL_26;
          goto LABEL_65;
        }
        if ( v6 == (_QWORD *)sub_B46EC0(v14, v18) )
          goto LABEL_25;
        ++v18;
      }
      if ( v6 == (_QWORD *)sub_B46EC0(v14, v18) )
        goto LABEL_25;
      ++v18;
LABEL_65:
      if ( v6 != (_QWORD *)sub_B46EC0(v14, v18) )
        goto LABEL_26;
      goto LABEL_25;
    }
  }
LABEL_26:
  v49 = a1 + 96;
  v11 = sub_B19060(a1 + 96, (__int64)v6, v9, v10);
  if ( (_BYTE)v11 )
    return 0;
  v19 = sub_AA4FF0((__int64)v6);
  if ( !v19 )
    goto LABEL_81;
  v20 = (unsigned int)*(unsigned __int8 *)(v19 - 24) - 39;
  if ( (unsigned int)v20 > 0x38 || (v21 = 0x100060000000001LL, !_bittest64(&v21, v20)) )
  {
    v22 = sub_AA4E30((__int64)v4);
    v23 = v6[2];
    v50 = v22;
    if ( v23 )
    {
      while ( 1 )
      {
        v24 = *(_QWORD *)(v23 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v24 - 30) <= 0xAu )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          return v11;
      }
      v51 = 0;
      v52 = 0;
      v54 = 0;
      v56 = v3;
LABEL_34:
      v25 = *(_QWORD *)(v24 + 40);
      v26 = *(_QWORD *)(v25 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v26 != v25 + 48 )
      {
        if ( v26 )
        {
          v27 = (unsigned int)*(unsigned __int8 *)(v26 - 24) - 30;
          if ( (unsigned int)v27 <= 0xA )
          {
            if ( *(_BYTE *)(v26 - 24) != 33 )
            {
              v28 = sub_27DCA00(a1, (__int64)v4, *(_QWORD *)(v24 + 40), a3, v50);
              v29 = v28;
              if ( v28 )
              {
                if ( *(_BYTE *)v28 == 17 )
                {
                  v30 = v28[8];
                  v31 = (__int64)(v28 + 6);
                  if ( v30 <= 0x40 )
                  {
                    v33 = *((_QWORD *)v28 + 3) == 0;
                  }
                  else
                  {
                    v47 = v28[8];
                    v48 = (char *)(v28 + 6);
                    v32 = sub_C444A0(v31);
                    v30 = v47;
                    v31 = (__int64)v48;
                    v33 = v47 == v32;
                  }
                  if ( v33 )
                  {
                    ++v54;
                    v57 = v25;
                  }
                  else
                  {
                    if ( v30 <= 0x40 )
                      v46 = *((_QWORD *)v29 + 3) == 1;
                    else
                      v46 = v30 - 1 == (unsigned int)sub_C444A0(v31);
                    if ( v46 )
                    {
                      ++v52;
                      v51 = v25;
                    }
                  }
                }
              }
            }
            while ( 1 )
            {
              v23 = *(_QWORD *)(v23 + 8);
              if ( !v23 )
                break;
              v24 = *(_QWORD *)(v23 + 24);
              if ( (unsigned __int8)(*(_BYTE *)v24 - 30) <= 0xAu )
                goto LABEL_34;
            }
            if ( v54 == 1 )
            {
              v35 = -64;
              goto LABEL_48;
            }
            if ( v52 == 1 )
            {
              v27 = v51;
              v34 = v57 == v51;
              v57 = v51;
              v24 = 32 * v34;
              v35 = -32 - v24;
LABEL_48:
              v36 = *(_QWORD *)(v56 + v35 - 24);
              if ( (!v36 || v4 != (_QWORD *)v36)
                && !(unsigned __int8)sub_B19060(v49, (__int64)v4, v24, v27)
                && !(unsigned __int8)sub_B19060(v49, v36, v37, v38) )
              {
                v39 = (unsigned __int8 *)sub_986580((__int64)v4);
                v40 = sub_27DC180(*(__int64 ***)(a1 + 24), v4, v39, *(_DWORD *)(a1 + 416));
                v41 = (unsigned __int8 *)sub_986580((__int64)v6);
                v42 = sub_27DC180(*(__int64 ***)(a1 + 24), v6, v41, *(_DWORD *)(a1 + 416));
                v43 = *(_DWORD *)(a1 + 416);
                v44 = v42;
                if ( v40 >= v42 )
                  v44 = v40;
                if ( v43 >= v44 && v43 >= v42 + v40 )
                {
                  v11 = 1;
                  sub_27E6080(a1, v57, (__int64)v6, (unsigned __int64)v4, v36);
                  return v11;
                }
              }
            }
            return 0;
          }
        }
      }
LABEL_81:
      BUG();
    }
  }
  return v11;
}
