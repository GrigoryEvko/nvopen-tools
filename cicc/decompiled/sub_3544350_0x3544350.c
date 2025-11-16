// Function: sub_3544350
// Address: 0x3544350
//
__int64 __fastcall sub_3544350(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v6; // r14d
  int v8; // r15d
  __int64 v9; // r14
  int v10; // esi
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int64 v14; // rax
  __int64 v15; // rbx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int64 v21; // rax
  char v22; // cl
  unsigned __int64 v23; // rdx
  char v24; // si
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rcx
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rax
  char v34; // cl
  unsigned __int64 v35; // rdx
  char v36; // si
  unsigned __int64 v37; // rcx
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rax
  unsigned __int64 v43; // rcx
  unsigned __int64 v44; // rax
  __int64 v45; // r12
  __int64 v46; // rbx
  unsigned __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  unsigned __int64 v49; // [rsp+8h] [rbp-88h]
  char v50; // [rsp+1Ah] [rbp-76h] BYREF
  char v51; // [rsp+1Bh] [rbp-75h] BYREF
  int v52; // [rsp+1Ch] [rbp-74h] BYREF
  int v53; // [rsp+20h] [rbp-70h] BYREF
  int v54; // [rsp+24h] [rbp-6Ch] BYREF
  int v55; // [rsp+28h] [rbp-68h] BYREF
  int v56; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v57; // [rsp+30h] [rbp-60h] BYREF
  char *v58; // [rsp+38h] [rbp-58h] BYREF
  __int64 v59; // [rsp+40h] [rbp-50h] BYREF
  __int64 v60; // [rsp+48h] [rbp-48h] BYREF
  __int64 v61; // [rsp+50h] [rbp-40h] BYREF
  char v62; // [rsp+58h] [rbp-38h]

  if ( !sub_3543BE0(a1, a2, (__int64)&v52) )
    return 1;
  if ( !sub_3543BE0(a1, a3, (__int64)&v53) )
    return 1;
  v8 = v52;
  if ( v52 != v53 )
    return 1;
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 32) + 16LL) + 200LL))(*(_QWORD *)(*(_QWORD *)(a1 + 32) + 16LL));
  if ( !(unsigned __int8)sub_2FE0930(*(__int64 **)(a1 + 16), a2, &v57, (__int64)&v59, (__int64)&v50, v9) )
    return 1;
  v6 = sub_2FE0930(*(__int64 **)(a1 + 16), a3, &v58, (__int64)&v60, (__int64)&v51, v9);
  if ( !(_BYTE)v6 || v50 || v51 )
    return 1;
  if ( (unsigned __int8)sub_2EAB6C0(v57, v58) )
    goto LABEL_22;
  if ( !*(_BYTE *)v57 && !*v58 )
  {
    v10 = *(_DWORD *)(v57 + 8);
    if ( v10 < 0 && *((int *)v58 + 2) < 0 )
    {
      v47 = sub_2EBEE10(*(_QWORD *)(a1 + 40), v10);
      v11 = sub_2EBEE10(*(_QWORD *)(a1 + 40), *((_DWORD *)v58 + 2));
      if ( v47 )
      {
        if ( v11
          && (*(_WORD *)(v47 + 68) == 68 || !*(_WORD *)(v47 + 68))
          && (*(_WORD *)(v11 + 68) == 68 || !*(_WORD *)(v11 + 68)) )
        {
          v12 = *(_QWORD *)(a1 + 904);
          v13 = v47;
          v48 = v11;
          v54 = 0;
          v55 = 0;
          v56 = 0;
          LODWORD(v61) = 0;
          sub_353CFA0(v13, v12, &v54, &v55);
          sub_353CFA0(v48, *(_QWORD *)(a1 + 904), &v56, &v61);
          v49 = sub_2EBEE10(*(_QWORD *)(a1 + 40), v54);
          v14 = sub_2EBEE10(*(_QWORD *)(a1 + 40), v56);
          if ( sub_2E88AF0(v49, v14, 0) )
          {
LABEL_22:
            v15 = -1;
            v16 = sub_2E864A0(a2);
            v17 = *(_QWORD *)v16;
            v18 = *(_QWORD *)(*(_QWORD *)v16 + 24LL);
            if ( (v18 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
            {
              v34 = *(_BYTE *)(v17 + 24);
              v35 = v18 >> 3;
              v36 = v34 & 2;
              if ( (v34 & 6) == 2 || (v34 & 1) != 0 )
              {
                v43 = HIDWORD(v18);
                v44 = HIWORD(v18);
                if ( !v36 )
                  LODWORD(v44) = v43;
                v15 = ((unsigned __int64)(unsigned int)v44 + 7) >> 3;
              }
              else
              {
                v37 = v18;
                v38 = v18;
                v39 = HIDWORD(v18);
                v40 = v37 >> 8;
                v41 = HIWORD(v38);
                if ( v36 )
                  LODWORD(v39) = v41;
                v42 = ((unsigned __int64)((unsigned __int16)v40 * (unsigned int)v39) + 7) >> 3;
                v15 = v42;
                if ( (v35 & 1) != 0 )
                  v15 = v42 | 0x4000000000000000LL;
              }
            }
            v19 = sub_2E864A0(a3);
            v20 = *(_QWORD *)v19;
            v21 = *(_QWORD *)(*(_QWORD *)v19 + 24LL);
            if ( (v21 & 0xFFFFFFFFFFFFFFF9LL) != 0 )
            {
              v22 = *(_BYTE *)(v20 + 24);
              v23 = v21 >> 3;
              v24 = v22 & 2;
              if ( (v22 & 6) == 2 || (v22 & 1) != 0 )
              {
                v32 = HIDWORD(v21);
                v33 = HIWORD(v21);
                if ( !v24 )
                  LODWORD(v33) = v32;
                v30 = ((unsigned __int64)(unsigned int)v33 + 7) >> 3;
              }
              else
              {
                v25 = v21;
                v26 = v21;
                v27 = HIWORD(v21);
                v28 = v25 >> 8;
                v29 = HIDWORD(v26);
                if ( !v24 )
                  LODWORD(v27) = v29;
                v30 = ((unsigned __int64)((unsigned __int16)v28 * (unsigned int)v27) + 7) >> 3;
                if ( (v23 & 1) != 0 )
                  v30 |= 0x4000000000000000uLL;
              }
              if ( v15 != -1 )
              {
                if ( v8 < 0 )
                {
                  v45 = v60 + v8;
                  v46 = v59;
                  v62 = v30 >> 62;
                  v61 = v30 & 0x3FFFFFFF;
                  LOBYTE(v6) = v46 <= v45 + sub_CA1930(&v61) - 1;
                }
                else
                {
                  v61 = v15 & 0x3FFFFFFFFFFFFFFFLL;
                  v62 = (v15 & 0x4000000000000000LL) != 0;
                  v31 = sub_CA1930(&v61);
                  LOBYTE(v6) = v59 + v31 - 1 >= v60 + v8;
                }
                return v6;
              }
            }
            return 1;
          }
        }
      }
    }
  }
  return v6;
}
