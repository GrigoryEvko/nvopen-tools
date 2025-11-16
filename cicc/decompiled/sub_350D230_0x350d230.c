// Function: sub_350D230
// Address: 0x350d230
//
__int64 __fastcall sub_350D230(_QWORD *a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  unsigned int v6; // r12d
  int v7; // eax
  __int64 v8; // r13
  unsigned int v9; // edx
  __int64 *v10; // rdi
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdi
  unsigned int v16; // ebx
  void (*v17)(); // rax
  __int64 v18; // r8
  __int64 v19; // r9
  _DWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rdx
  _BYTE *v25; // r13
  bool v26; // r8
  bool v27; // zf
  _QWORD *v28; // r12
  _BYTE *v29; // r15
  bool v30; // bl
  __int64 v31; // rdi
  void (*v32)(); // rax
  __int64 v33; // r14
  unsigned __int64 v34; // rsi
  int v36; // edi
  int v37; // r10d
  __int64 v38; // [rsp+0h] [rbp-130h]
  unsigned int v39; // [rsp+18h] [rbp-118h]
  int v40; // [rsp+1Ch] [rbp-114h]
  __int64 v41; // [rsp+20h] [rbp-110h]
  __int64 v42; // [rsp+28h] [rbp-108h]
  _DWORD *v44; // [rsp+38h] [rbp-F8h]
  _BYTE *v45; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v46; // [rsp+48h] [rbp-E8h]
  _BYTE v47[64]; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v48; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v49; // [rsp+98h] [rbp-98h]
  __int64 v50; // [rsp+A0h] [rbp-90h]
  __int64 v51; // [rsp+A8h] [rbp-88h]
  _BYTE *v52; // [rsp+B0h] [rbp-80h]
  __int64 v53; // [rsp+B8h] [rbp-78h]
  _BYTE v54[112]; // [rsp+C0h] [rbp-70h] BYREF

  v52 = v54;
  v53 = 0x800000000LL;
  v42 = 4 * a4;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v41 = (4 * a4) >> 4;
  v44 = (_DWORD *)((char *)a3 + ((4 * a4) & 0xFFFFFFFFFFFFFFF0LL));
  while ( 1 )
  {
LABEL_2:
    while ( 1 )
    {
      v6 = *(_DWORD *)(a2 + 8);
      if ( !v6 )
        break;
      v34 = *(_QWORD *)(*(_QWORD *)a2 + 8LL * v6 - 8);
      *(_DWORD *)(a2 + 8) = v6 - 1;
      sub_350BDD0(a1, v34, (__int64)&v48);
    }
    v7 = v53;
    if ( !(_DWORD)v53 )
      break;
    v8 = *(_QWORD *)&v52[8 * (unsigned int)v53 - 8];
    if ( (_DWORD)v51 )
    {
      v9 = (v51 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v10 = (__int64 *)(v49 + 8LL * v9);
      v11 = *v10;
      if ( v8 == *v10 )
      {
LABEL_6:
        *v10 = -8192;
        v7 = v53;
        LODWORD(v50) = v50 - 1;
        ++HIDWORD(v50);
      }
      else
      {
        v36 = 1;
        while ( v11 != -4096 )
        {
          v37 = v36 + 1;
          v9 = (v51 - 1) & (v36 + v9);
          v10 = (__int64 *)(v49 + 8LL * v9);
          v11 = *v10;
          if ( v8 == *v10 )
            goto LABEL_6;
          v36 = v37;
        }
      }
    }
    LODWORD(v53) = v7 - 1;
    if ( !(unsigned __int8)sub_350B7E0((__int64)a1, v8, a2) )
    {
      v15 = a1[7];
      v16 = *(_DWORD *)(v8 + 112);
      if ( v15 )
      {
        v17 = *(void (**)())(*(_QWORD *)v15 + 40LL);
        if ( v17 != nullsub_1659 )
          ((void (__fastcall *)(__int64, _QWORD))v17)(v15, v16);
      }
      if ( (unsigned __int8)sub_2E168A0((_QWORD *)a1[4], v8, a2, v12, v13, v14) )
      {
        v20 = a3;
        v21 = (__int64)&a3[(unsigned __int64)v42 / 4];
        v22 = v42 >> 2;
        if ( v41 > 0 )
        {
          v22 = (__int64)v44;
          while ( v16 != *v20 )
          {
            if ( v16 == v20[1] )
            {
              if ( (_DWORD *)v21 != v20 + 1 )
                goto LABEL_2;
              goto LABEL_20;
            }
            if ( v16 == v20[2] )
            {
              if ( (_DWORD *)v21 != v20 + 2 )
                goto LABEL_2;
              goto LABEL_20;
            }
            if ( v16 == v20[3] )
            {
              if ( (_DWORD *)v21 != v20 + 3 )
                goto LABEL_2;
              goto LABEL_20;
            }
            v20 += 4;
            if ( v44 == v20 )
            {
              v22 = (v21 - (__int64)v44) >> 2;
              goto LABEL_39;
            }
          }
          goto LABEL_19;
        }
LABEL_39:
        if ( v22 == 2 )
          goto LABEL_56;
        if ( v22 == 3 )
        {
          if ( v16 != *v20 )
          {
            ++v20;
LABEL_56:
            if ( v16 == *v20 )
              goto LABEL_19;
            if ( v16 == *++v20 )
              goto LABEL_19;
            goto LABEL_20;
          }
LABEL_19:
          if ( (_DWORD *)v21 == v20 )
            goto LABEL_20;
        }
        else
        {
          if ( v22 == 1 && v16 == *v20 )
            goto LABEL_19;
LABEL_20:
          sub_2E0A330((__int64 *)v8, v21, (char *)v22, v42 >> 2, v18, v19);
          v23 = a1[4];
          v45 = v47;
          v46 = 0x800000000LL;
          sub_2E15100(v23, v8, (__int64)&v45);
          v24 = a1[5];
          if ( v24 )
          {
            v6 = *(_DWORD *)(*(_QWORD *)(v24 + 80) + 4LL * (v16 & 0x7FFFFFFF));
            if ( !v6 )
              v6 = v16;
          }
          v25 = v45;
          if ( &v45[8 * (unsigned int)v46] != v45 )
          {
            v40 = v6;
            v26 = v6 != v16;
            v27 = v6 == 0;
            v39 = v16;
            v28 = a1;
            v38 = a2;
            v29 = &v45[8 * (unsigned int)v46];
            v30 = !v27 && v26;
            do
            {
              while ( 1 )
              {
                v33 = *(_QWORD *)v25;
                if ( v30 )
                  sub_350ABD0(v28[5], *(_DWORD *)(v33 + 112), v40);
                v31 = v28[7];
                if ( v31 )
                {
                  v32 = *(void (**)())(*(_QWORD *)v31 + 48LL);
                  if ( v32 != nullsub_1660 )
                    break;
                }
                v25 += 8;
                if ( v29 == v25 )
                  goto LABEL_32;
              }
              v25 += 8;
              ((void (__fastcall *)(__int64, _QWORD, _QWORD))v32)(v31, *(unsigned int *)(v33 + 112), v39);
            }
            while ( v29 != v25 );
LABEL_32:
            a2 = v38;
            v25 = v45;
            a1 = v28;
          }
          if ( v25 != v47 )
            _libc_free((unsigned __int64)v25);
        }
      }
    }
  }
  if ( v52 != v54 )
    _libc_free((unsigned __int64)v52);
  return sub_C7D6A0(v49, 8LL * (unsigned int)v51, 8);
}
