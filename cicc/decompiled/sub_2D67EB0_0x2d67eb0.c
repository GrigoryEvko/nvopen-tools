// Function: sub_2D67EB0
// Address: 0x2d67eb0
//
__int64 __fastcall sub_2D67EB0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // r13
  unsigned __int64 v4; // rax
  unsigned __int8 v6; // r15
  __int64 v7; // rax
  int v8; // edx
  __int64 v9; // r8
  __int64 v10; // r10
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r11
  unsigned int v15; // ebx
  __int64 v16; // r14
  __int64 v18; // r15
  __int64 v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rax
  int v22; // edx
  _BYTE *v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // r8
  __int64 *v26; // r9
  _BYTE *v27; // r12
  __int64 v28; // r15
  __int64 *v29; // r14
  __int64 v30; // rax
  int v31; // edx
  __int64 *v32; // r9
  __int64 v33; // r8
  unsigned int v34; // r15d
  _BYTE *v35; // r15
  __int64 v36; // r14
  __int64 v37; // r12
  int v38; // eax
  __int64 *v39; // [rsp+10h] [rbp-90h]
  unsigned __int8 v40; // [rsp+10h] [rbp-90h]
  __int64 *v41; // [rsp+10h] [rbp-90h]
  unsigned __int8 v42; // [rsp+18h] [rbp-88h]
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+20h] [rbp-80h]
  _BYTE *v45; // [rsp+20h] [rbp-80h]
  __int64 v46; // [rsp+28h] [rbp-78h]
  _BYTE *v47; // [rsp+28h] [rbp-78h]
  __int64 v48; // [rsp+28h] [rbp-78h]
  __int64 v49; // [rsp+30h] [rbp-70h] BYREF
  int v50; // [rsp+38h] [rbp-68h]
  __int64 v51; // [rsp+40h] [rbp-60h] BYREF
  int v52; // [rsp+48h] [rbp-58h]
  _BYTE *v53; // [rsp+50h] [rbp-50h] BYREF
  _BYTE *v54; // [rsp+58h] [rbp-48h]
  _BYTE *v55; // [rsp+60h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_QWORD *)(v3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v4 == v3 + 48 )
    goto LABEL_58;
  if ( !v4 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_58:
    BUG();
  if ( *(_BYTE *)(v4 - 24) == 33
    && (v6 = sub_2D583C0(a1)) != 0
    && ((v43 = *(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), v7 = sub_DFAFF0(a2), v9 = v43, v8)
      ? (LOBYTE(v2) = v8 > 0)
      : (LOBYTE(v2) = v7 > 1),
        !(_BYTE)v2
     && (v10 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)), *(_BYTE *)v10 > 0x1Cu)
     && v3 == *(_QWORD *)(v10 + 40)) )
  {
    v11 = *(_QWORD *)(a1 + 16);
    if ( v11 )
    {
      while ( 1 )
      {
        v12 = *(_QWORD *)(v11 + 24);
        if ( *(_BYTE *)v12 > 0x1Cu && v3 != *(_QWORD *)(v12 + 40) )
          break;
        v11 = *(_QWORD *)(v11 + 8);
        if ( !v11 )
          return v2;
      }
      v55 = 0;
      v13 = *(_QWORD *)(v10 + 16);
      v53 = 0;
      v54 = 0;
      if ( v13 )
      {
        v39 = (__int64 *)(v43 + 24);
        v15 = v2;
        v44 = v10;
        v16 = v3;
        v46 = v9;
        v42 = v6;
        v18 = v13;
        do
        {
          v19 = *(_QWORD *)(v18 + 24);
          if ( a1 != v19 )
          {
            if ( *(_BYTE *)v19 <= 0x1Cu )
              goto LABEL_22;
            if ( *(_QWORD *)(v19 + 40) != v16 )
            {
              if ( *(_BYTE *)v19 != 63
                || (v51 = *(_QWORD *)(v18 + 24), !(unsigned __int8)sub_2D583C0(v19))
                || (v20 = *(_QWORD *)(v51 - 32LL * (*(_DWORD *)(v51 + 4) & 0x7FFFFFF))) == 0
                || v44 != v20
                || *(_QWORD *)(a1 + 72) != *(_QWORD *)(v51 + 72)
                || *(_QWORD *)(*(_QWORD *)(v51 + 32 * (1LL - (*(_DWORD *)(v51 + 4) & 0x7FFFFFF))) + 8LL) != *(_QWORD *)(v46 + 8)
                || ((v21 = sub_DFAFF0(a2), !v22) ? (v22 = v21 > 1) : (LOBYTE(v22) = v22 > 0), (_BYTE)v22) )
              {
LABEL_22:
                v2 = v15;
                goto LABEL_23;
              }
              v23 = v54;
              if ( v54 == v55 )
              {
                sub_2CF6420((__int64)&v53, v54, &v51);
              }
              else
              {
                if ( v54 )
                {
                  *(_QWORD *)v54 = v51;
                  v23 = v54;
                }
                v54 = v23 + 8;
              }
            }
          }
          v18 = *(_QWORD *)(v18 + 8);
        }
        while ( v18 );
        v2 = v15;
        v24 = a1;
        v25 = v46;
        v47 = v54;
        v26 = v39;
        if ( v54 != v53 )
        {
          v27 = v53;
          v28 = v25;
          v40 = v2;
          v29 = v26;
          do
          {
            sub_9865C0(
              (__int64)&v51,
              *(_QWORD *)(*(_QWORD *)v27 + 32 * (1LL - (*(_DWORD *)(*(_QWORD *)v27 + 4LL) & 0x7FFFFFF))) + 24LL);
            sub_C46B40((__int64)&v51, v29);
            v50 = v52;
            v49 = v51;
            v30 = sub_DFAFF0(a2);
            if ( v31 )
            {
              if ( v31 > 0 )
                goto LABEL_47;
            }
            else if ( v30 > 1 )
            {
LABEL_47:
              v2 = v40;
              sub_969240(&v49);
              goto LABEL_23;
            }
            v27 += 8;
            sub_969240(&v49);
          }
          while ( v47 != v27 );
          v32 = v29;
          v33 = v28;
          v34 = v42;
          v45 = v54;
          if ( v54 != v53 )
          {
            v35 = v53;
            v36 = v33;
            v41 = v32;
            do
            {
              v37 = *(_QWORD *)v35;
              sub_AC2B30(*(_QWORD *)v35 - 32LL * (*(_DWORD *)(*(_QWORD *)v35 + 4LL) & 0x7FFFFFF), v24);
              sub_9865C0((__int64)&v49, *(_QWORD *)(v37 + 32 * (1LL - (*(_DWORD *)(v37 + 4) & 0x7FFFFFF))) + 24LL);
              sub_C46B40((__int64)&v49, v41);
              v38 = v50;
              v50 = 0;
              v52 = v38;
              v51 = v49;
              v48 = sub_AD8D80(*(_QWORD *)(v36 + 8), (__int64)&v51);
              sub_969240(&v51);
              sub_969240(&v49);
              sub_AC2B30(v37 + 32 * (1LL - (*(_DWORD *)(v37 + 4) & 0x7FFFFFF)), v48);
              if ( !sub_B4DE30(v24) )
                sub_B4DE00(v37, 0);
              v35 += 8;
            }
            while ( v45 != v35 );
            v34 = v42;
          }
          v2 = v34;
        }
LABEL_23:
        if ( v53 )
          j_j___libc_free_0((unsigned __int64)v53);
      }
    }
  }
  else
  {
    return 0;
  }
  return v2;
}
