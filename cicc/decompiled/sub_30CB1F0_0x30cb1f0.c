// Function: sub_30CB1F0
// Address: 0x30cb1f0
//
__int64 *__fastcall sub_30CB1F0(__int64 *a1, __int64 a2, int *a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  int v5; // ebx
  unsigned __int8 v6; // al
  unsigned __int8 **v7; // rdx
  unsigned __int64 v8; // rbx
  char v9; // r13
  unsigned __int8 v10; // al
  _BYTE **v11; // rdx
  _BYTE *v12; // rdi
  unsigned int v13; // eax
  unsigned __int8 *v14; // rax
  unsigned __int8 v15; // dl
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rdx
  _BYTE *v19; // rsi
  size_t v20; // rdx
  unsigned __int8 *v21; // rsi
  __int64 v22; // r8
  _BYTE *v23; // rax
  char *v24; // rsi
  int v25; // eax
  unsigned __int8 v26; // al
  unsigned __int64 *v27; // rax
  unsigned __int64 v29; // rax
  unsigned __int8 v30; // al
  unsigned __int8 **v31; // rdx
  unsigned __int8 *v32; // rax
  unsigned __int8 v33; // dl
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  _WORD *v37; // rdx
  _QWORD *v38; // rbx
  unsigned __int64 v39; // rcx
  char *v40; // rsi
  unsigned __int64 v41; // rax
  _QWORD *v42; // r13
  unsigned __int64 v43; // rcx
  _BYTE *v44; // rsi
  unsigned __int64 v45; // rax
  unsigned int v48; // [rsp+34h] [rbp-11Ch]
  __int64 v49; // [rsp+48h] [rbp-108h]
  __int64 v50; // [rsp+50h] [rbp-100h]
  char v51; // [rsp+74h] [rbp-DCh] BYREF
  _BYTE v52[11]; // [rsp+75h] [rbp-DBh] BYREF
  unsigned __int64 v53[2]; // [rsp+80h] [rbp-D0h] BYREF
  _BYTE v54[16]; // [rsp+90h] [rbp-C0h] BYREF
  unsigned __int8 *v55; // [rsp+A0h] [rbp-B0h] BYREF
  size_t v56; // [rsp+A8h] [rbp-A8h]
  _BYTE v57[4]; // [rsp+B0h] [rbp-A0h] BYREF
  char v58; // [rsp+B4h] [rbp-9Ch] BYREF
  _BYTE v59[11]; // [rsp+B5h] [rbp-9Bh] BYREF
  unsigned __int8 *v60; // [rsp+C0h] [rbp-90h] BYREF
  size_t v61; // [rsp+C8h] [rbp-88h]
  _QWORD v62[2]; // [rsp+D0h] [rbp-80h] BYREF
  _QWORD v63[3]; // [rsp+E0h] [rbp-70h] BYREF
  _BYTE *v64; // [rsp+F8h] [rbp-58h]
  _BYTE *v65; // [rsp+100h] [rbp-50h]
  __int64 v66; // [rsp+108h] [rbp-48h]
  unsigned __int64 *v67; // [rsp+110h] [rbp-40h]

  v66 = 0x100000000LL;
  v53[0] = (unsigned __int64)v54;
  v63[0] = &unk_49DD210;
  v53[1] = 0;
  v54[0] = 0;
  v63[1] = 0;
  v63[2] = 0;
  v64 = 0;
  v65 = 0;
  v67 = v53;
  sub_CB5980((__int64)v63, 0, 0, 0);
  v3 = sub_B10CD0(a2);
  if ( v3 )
  {
    v4 = v3;
    while ( 1 )
    {
      v50 = v4 - 16;
      v6 = *(_BYTE *)(v4 - 16);
      if ( (v6 & 2) != 0 )
        v7 = *(unsigned __int8 ***)(v4 - 32);
      else
        v7 = (unsigned __int8 **)(v50 - 8LL * ((v6 >> 2) & 0xF));
      v5 = *(_DWORD *)(v4 + 4);
      v8 = (unsigned int)(v5 - *((_DWORD *)sub_AF34D0(*v7) + 4));
      v9 = qword_4F813A8[8];
      v10 = *(_BYTE *)(v4 - 16);
      if ( (v10 & 2) != 0 )
      {
        v11 = *(_BYTE ***)(v4 - 32);
        v12 = *v11;
        if ( **v11 == 20 )
          goto LABEL_7;
      }
      else
      {
        v12 = *(_BYTE **)(v50 - 8LL * ((v10 >> 2) & 0xF));
        if ( *v12 == 20 )
        {
LABEL_7:
          v13 = *((_DWORD *)v12 + 1);
          if ( (v13 & 7) == 7 && (v13 & 0xFFFFFFF8) != 0 )
          {
            if ( (v13 & 0x10000000) != 0 )
              v48 = HIWORD(v13) & 7;
            else
              v48 = (unsigned __int16)(v13 >> 3);
            goto LABEL_11;
          }
          if ( LOBYTE(qword_4F813A8[8]) )
          {
            v48 = (unsigned __int8)v13;
            v9 = (unsigned __int8)v13 != 0;
          }
          else
          {
            v48 = 0;
            if ( (v13 & 1) == 0 )
            {
              v48 = (v13 >> 1) & 0x1F;
              if ( ((v13 >> 1) & 0x20) == 0 )
                goto LABEL_11;
              v48 = (v13 >> 2) & 0xFE0 | (v13 >> 1) & 0x1F;
              v9 = v48 != 0;
            }
          }
          goto LABEL_12;
        }
      }
      if ( !LOBYTE(qword_4F813A8[8]) )
      {
        v48 = 0;
LABEL_11:
        v9 = v48 != 0;
        goto LABEL_12;
      }
      v48 = 0;
      v9 = 0;
LABEL_12:
      v14 = sub_AF34D0(v12);
      v15 = *(v14 - 16);
      if ( (v15 & 2) != 0 )
      {
        v16 = *(_QWORD *)(*((_QWORD *)v14 - 4) + 24LL);
        if ( !v16 )
          goto LABEL_38;
      }
      else
      {
        v16 = *(_QWORD *)&v14[-8 * ((v15 >> 2) & 0xF) + 8];
        if ( !v16 )
          goto LABEL_38;
      }
      v17 = sub_B91420(v16);
      v19 = (_BYTE *)v17;
      if ( v18 )
      {
        if ( v17 )
          goto LABEL_16;
        goto LABEL_44;
      }
LABEL_38:
      v30 = *(_BYTE *)(v4 - 16);
      if ( (v30 & 2) != 0 )
        v31 = *(unsigned __int8 ***)(v4 - 32);
      else
        v31 = (unsigned __int8 **)(v50 - 8LL * ((v30 >> 2) & 0xF));
      v32 = sub_AF34D0(*v31);
      v33 = *(v32 - 16);
      if ( (v33 & 2) != 0 )
        v34 = *((_QWORD *)v32 - 4);
      else
        v34 = (__int64)&v32[-8 * ((v33 >> 2) & 0xF) - 16];
      v35 = *(_QWORD *)(v34 + 16);
      if ( v35 )
      {
        v19 = (_BYTE *)sub_B91420(v35);
        if ( v19 )
        {
LABEL_16:
          v55 = v57;
          sub_30CA380((__int64 *)&v55, v19, (__int64)&v19[v18]);
          v20 = v56;
          v21 = v55;
          goto LABEL_17;
        }
      }
LABEL_44:
      v57[0] = 0;
      v20 = 0;
      v56 = 0;
      v21 = v57;
      v55 = v57;
LABEL_17:
      v22 = sub_CB6200((__int64)v63, v21, v20);
      v23 = *(_BYTE **)(v22 + 32);
      if ( *(_BYTE **)(v22 + 24) == v23 )
      {
        v22 = sub_CB6200(v22, (unsigned __int8 *)":", 1u);
      }
      else
      {
        *v23 = 58;
        ++*(_QWORD *)(v22 + 32);
      }
      if ( v8 )
      {
        v24 = v52;
        do
        {
          *--v24 = v8 % 0xA + 48;
          v29 = v8;
          v8 /= 0xAu;
        }
        while ( v29 > 9 );
      }
      else
      {
        v51 = 48;
        v24 = &v51;
      }
      v49 = v22;
      v60 = (unsigned __int8 *)v62;
      sub_30CA4D0((__int64 *)&v60, v24, (__int64)v52);
      sub_CB6200(v49, v60, v61);
      if ( v60 != (unsigned __int8 *)v62 )
        j_j___libc_free_0((unsigned __int64)v60);
      if ( v55 != v57 )
        j_j___libc_free_0((unsigned __int64)v55);
      v25 = *a3;
      if ( (*a3 & 0xFFFFFFFD) == 1 )
      {
        if ( v64 == v65 )
        {
          v38 = (_QWORD *)sub_CB6200((__int64)v63, (unsigned __int8 *)":", 1u);
        }
        else
        {
          *v65 = 58;
          v38 = v63;
          ++v65;
        }
        v39 = *(unsigned __int16 *)(v4 + 2);
        if ( *(_WORD *)(v4 + 2) )
        {
          v40 = v59;
          do
          {
            *--v40 = v39 % 0xA + 48;
            v41 = v39;
            v39 /= 0xAu;
          }
          while ( v41 > 9 );
        }
        else
        {
          v58 = 48;
          v40 = &v58;
        }
        v60 = (unsigned __int8 *)v62;
        sub_30CA4D0((__int64 *)&v60, v40, (__int64)v59);
        sub_CB6200((__int64)v38, v60, v61);
        if ( v60 != (unsigned __int8 *)v62 )
          j_j___libc_free_0((unsigned __int64)v60);
        v25 = *a3;
      }
      if ( (unsigned int)(v25 - 2) <= 1 && v9 )
      {
        if ( v64 == v65 )
        {
          v42 = (_QWORD *)sub_CB6200((__int64)v63, (unsigned __int8 *)".", 1u);
        }
        else
        {
          *v65 = 46;
          v42 = v63;
          ++v65;
        }
        v43 = v48;
        v44 = v59;
        do
        {
          *--v44 = v43 % 0xA + 48;
          v45 = v43;
          v43 /= 0xAu;
        }
        while ( v45 > 9 );
        v60 = (unsigned __int8 *)v62;
        sub_30CA4D0((__int64 *)&v60, v44, (__int64)v59);
        sub_CB6200((__int64)v42, v60, v61);
        if ( v60 != (unsigned __int8 *)v62 )
          j_j___libc_free_0((unsigned __int64)v60);
      }
      v26 = *(_BYTE *)(v4 - 16);
      if ( (v26 & 2) != 0 )
      {
        if ( *(_DWORD *)(v4 - 24) != 2 )
          break;
        v36 = *(_QWORD *)(v4 - 32);
      }
      else
      {
        if ( ((*(_WORD *)(v4 - 16) >> 6) & 0xF) != 2 )
          break;
        v36 = v50 - 8LL * ((v26 >> 2) & 0xF);
      }
      v4 = *(_QWORD *)(v36 + 8);
      if ( !v4 )
        break;
      v37 = v65;
      if ( (unsigned __int64)(v64 - v65) <= 2 )
      {
        sub_CB6200((__int64)v63, " @ ", 3u);
      }
      else
      {
        v65[2] = 32;
        *v37 = 16416;
        v65 += 3;
      }
    }
  }
  v27 = v67;
  *a1 = (__int64)(a1 + 2);
  sub_30CA4D0(a1, (_BYTE *)*v27, *v27 + v27[1]);
  v63[0] = &unk_49DD210;
  sub_CB5840((__int64)v63);
  if ( (_BYTE *)v53[0] != v54 )
    j_j___libc_free_0(v53[0]);
  return a1;
}
