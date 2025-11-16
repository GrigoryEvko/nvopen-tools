// Function: sub_1AC5690
// Address: 0x1ac5690
//
__int64 __fastcall sub_1AC5690(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v3; // rdi
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  char v6; // dl
  bool v7; // zf
  __int64 v8; // r14
  __int64 v9; // r12
  __int64 i; // r14
  int v11; // r8d
  int v12; // r9d
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r14
  _QWORD *v17; // r13
  _BYTE *v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r12
  __int64 v21; // r14
  _QWORD **v22; // rax
  _QWORD *v23; // rdi
  __int64 v24; // r14
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // r14
  _QWORD *v30; // rax
  _QWORD *v31; // rbx
  __int64 v32; // r14
  __int64 v33; // rdi
  __int64 v34; // r12
  __int64 v35; // rsi
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  _BYTE *v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rsi
  unsigned __int8 *v41; // rsi
  _QWORD *v43; // r13
  _QWORD *v44; // r10
  __int64 *v45; // rax
  const char *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // r13
  _QWORD *v49; // [rsp+0h] [rbp-150h]
  int v50; // [rsp+Ch] [rbp-144h]
  __int64 v51; // [rsp+18h] [rbp-138h]
  __int64 v52; // [rsp+18h] [rbp-138h]
  _QWORD *v53; // [rsp+20h] [rbp-130h] BYREF
  __int16 v54; // [rsp+30h] [rbp-120h]
  unsigned __int64 v55[2]; // [rsp+40h] [rbp-110h] BYREF
  _QWORD v56[8]; // [rsp+50h] [rbp-100h] BYREF
  _BYTE *v57; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v58; // [rsp+98h] [rbp-B8h]
  _BYTE v59[176]; // [rsp+A0h] [rbp-B0h] BYREF

  if ( *(_BYTE *)(a1 + 104) )
    return 0;
  v1 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 16);
  if ( v1 == v3 )
  {
LABEL_6:
    v7 = *(_BYTE *)(a1 + 105) == 0;
    *(_BYTE *)(a1 + 104) = 1;
    if ( !v7 && !(unsigned __int8)sub_1560180(*(_QWORD *)a1 + 112LL, 30) )
    {
      v8 = *(_QWORD *)a1;
      v57 = v59;
      v58 = 0x1000000000LL;
      v9 = *(_QWORD *)(v8 + 80);
      v51 = v8 + 72;
      if ( v8 + 72 != v9 )
      {
        do
        {
          if ( !v9 )
            BUG();
          for ( i = *(_QWORD *)(v9 + 24); v9 + 16 != i; i = *(_QWORD *)(i + 8) )
          {
            if ( !i )
              BUG();
            if ( *(_BYTE *)(i - 8) == 78 && !(unsigned __int8)sub_1560260((_QWORD *)(i + 32), -1, 30) )
            {
              v13 = *(_QWORD *)(i - 48);
              if ( *(_BYTE *)(v13 + 16) || (v55[0] = *(_QWORD *)(v13 + 112), !(unsigned __int8)sub_1560260(v55, -1, 30)) )
              {
                v14 = (unsigned int)v58;
                if ( (unsigned int)v58 >= HIDWORD(v58) )
                {
                  sub_16CD150((__int64)&v57, v59, 0, 8, v11, v12);
                  v14 = (unsigned int)v58;
                }
                *(_QWORD *)&v57[8 * v14] = i - 24;
                LODWORD(v58) = v58 + 1;
              }
            }
          }
          v9 = *(_QWORD *)(v9 + 8);
        }
        while ( v51 != v9 );
        if ( !(_DWORD)v58 )
        {
          v38 = v57;
          v34 = 0;
LABEL_41:
          if ( v38 != v59 )
            _libc_free((unsigned __int64)v38);
          return v34;
        }
        v15 = sub_15E0530(*(_QWORD *)a1);
        v16 = *(_QWORD *)a1;
        v17 = (_QWORD *)v15;
        v18 = *(_BYTE **)(a1 + 8);
        LOWORD(v56[0]) = 257;
        if ( *v18 )
        {
          v55[0] = (unsigned __int64)v18;
          LOBYTE(v56[0]) = 3;
        }
        v19 = (_QWORD *)sub_22077B0(64);
        v20 = (__int64)v19;
        if ( v19 )
          sub_157FB60(v19, (__int64)v17, (__int64)v55, v16, 0);
        v21 = sub_1643350(v17);
        v22 = (_QWORD **)sub_16471D0(v17, 0);
        v23 = *v22;
        v56[0] = v22;
        v56[1] = v21;
        v55[0] = (unsigned __int64)v56;
        v55[1] = 0x800000002LL;
        v24 = sub_1645600(v23, v56, 2, 0);
        if ( (_QWORD *)v55[0] != v56 )
          _libc_free(v55[0]);
        v25 = *(_QWORD *)a1;
        if ( (*(_BYTE *)(*(_QWORD *)a1 + 18LL) & 8) == 0 )
        {
          v43 = *(_QWORD **)(v25 + 40);
          v44 = (_QWORD *)*v43;
          v54 = 260;
          v53 = v43 + 30;
          v49 = v44;
          sub_16E1010((__int64)v55, (__int64)&v53);
          v50 = sub_14DDD70(v55);
          v45 = (__int64 *)sub_1643350(v49);
          v52 = sub_16453E0(v45, 1u);
          v46 = sub_14DDC90(v50);
          v48 = sub_1632190((__int64)v43, (__int64)v46, v47, v52);
          if ( (_QWORD *)v55[0] != v56 )
            j_j___libc_free_0(v55[0], v56[0] + 1LL);
          sub_15E3D80(*(_QWORD *)a1, v48);
          v25 = *(_QWORD *)a1;
        }
        v26 = sub_15E38F0(v25);
        v27 = sub_14DD7D0(v26);
        if ( v27 > 10 )
        {
          if ( v27 != 12 )
          {
LABEL_31:
            v55[0] = (unsigned __int64)"cleanup.lpad";
            LOWORD(v56[0]) = 259;
            v28 = sub_15F59C0(v24, 1u, (__int64)v55, v20);
            *(_WORD *)(v28 + 18) |= 1u;
            v29 = v28;
            v30 = sub_1648A60(56, 1u);
            v31 = v30;
            if ( v30 )
              sub_15F7350((__int64)v30, v29, v20);
            v32 = 8LL * (unsigned int)(v58 - 1);
            if ( (_DWORD)v58 )
            {
              do
              {
                v33 = *(_QWORD *)&v57[v32];
                v32 -= 8;
                sub_1AEFCD0(v33, v20);
              }
              while ( v32 != -8 );
            }
            v34 = a1 + 32;
            *(_QWORD *)(a1 + 40) = v31[5];
            *(_QWORD *)(a1 + 48) = v31 + 3;
            v35 = v31[6];
            v55[0] = v35;
            if ( v35 )
            {
              sub_1623A60((__int64)v55, v35, 2);
              v36 = *(_QWORD *)(a1 + 32);
              if ( !v36 )
                goto LABEL_38;
            }
            else
            {
              v36 = *(_QWORD *)(a1 + 32);
              if ( !v36 )
              {
LABEL_40:
                v38 = v57;
                goto LABEL_41;
              }
            }
            sub_161E7C0(a1 + 32, v36);
LABEL_38:
            v37 = (unsigned __int8 *)v55[0];
            *(_QWORD *)(a1 + 32) = v55[0];
            if ( v37 )
              sub_1623210((__int64)v55, v37, a1 + 32);
            goto LABEL_40;
          }
        }
        else if ( v27 <= 6 )
        {
          goto LABEL_31;
        }
        sub_16BD130("Scoped EH not supported", 1u);
      }
    }
    return 0;
  }
  while ( 1 )
  {
    v4 = *(_QWORD *)(v3 + 8);
    *(_QWORD *)(a1 + 16) = v4;
    v5 = sub_157EBA0(v3 - 24);
    v6 = *(_BYTE *)(v5 + 16);
    if ( v6 == 25 || v6 == 30 )
      break;
    v3 = v4;
    if ( v1 == v4 )
      goto LABEL_6;
  }
  v34 = a1 + 32;
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(v5 + 40);
  *(_QWORD *)(a1 + 48) = v5 + 24;
  v39 = *(_QWORD *)(v5 + 48);
  v57 = (_BYTE *)v39;
  if ( v39 )
  {
    sub_1623A60((__int64)&v57, v39, 2);
    v40 = *(_QWORD *)(a1 + 32);
    if ( !v40 )
      goto LABEL_46;
  }
  else
  {
    v40 = *(_QWORD *)(a1 + 32);
    if ( !v40 )
      return v34;
  }
  sub_161E7C0(a1 + 32, v40);
LABEL_46:
  v41 = v57;
  *(_QWORD *)(a1 + 32) = v57;
  if ( v41 )
    sub_1623210((__int64)&v57, v41, a1 + 32);
  return v34;
}
