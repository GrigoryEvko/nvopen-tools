// Function: sub_10BE880
// Address: 0x10be880
//
_QWORD *__fastcall sub_10BE880(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, char a6)
{
  unsigned int v9; // ebx
  unsigned int v10; // r13d
  __int64 v11; // rsi
  unsigned int v12; // ecx
  __int64 *v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // rbx
  unsigned int v16; // eax
  __int64 v17; // r12
  __int64 *v18; // r15
  _QWORD *v19; // r14
  _QWORD **v20; // rdx
  int v21; // ecx
  __int64 *v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rbx
  __int64 v25; // r12
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 *v28; // rbx
  __int64 v29; // r15
  _QWORD **v31; // rdx
  int v32; // ecx
  __int64 *v33; // rax
  __int64 v34; // rsi
  __int64 v35; // r12
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rdx
  unsigned int v39; // esi
  __int64 v40; // r13
  __int64 v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 *v44; // [rsp+0h] [rbp-D0h]
  __int64 v45; // [rsp+8h] [rbp-C8h]
  unsigned int v46; // [rsp+8h] [rbp-C8h]
  __int64 v47; // [rsp+28h] [rbp-A8h]
  __int64 v48; // [rsp+30h] [rbp-A0h]
  __int64 v49; // [rsp+38h] [rbp-98h]
  const char *v50; // [rsp+40h] [rbp-90h] BYREF
  __int64 v51; // [rsp+48h] [rbp-88h]
  const char *v52; // [rsp+50h] [rbp-80h]
  __int16 v53; // [rsp+60h] [rbp-70h]
  const char *v54; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v55; // [rsp+78h] [rbp-58h]
  __int16 v56; // [rsp+90h] [rbp-40h]

  v9 = *(_DWORD *)(a3 + 8);
  v10 = 35 - ((a6 == 0) - 1);
  v47 = *(_QWORD *)(a2 + 8);
  if ( a5 )
  {
    v11 = *(_QWORD *)a3;
    v12 = v9 - 1;
    if ( v9 <= 0x40 )
    {
      if ( 1LL << v12 == v11 )
      {
LABEL_28:
        v10 = sub_B52E90(v10);
LABEL_23:
        v53 = 257;
        v28 = *(__int64 **)(a1 + 32);
        v29 = sub_AD8D80(v47, a4);
        v19 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v28[10] + 56LL))(
                          v28[10],
                          v10,
                          a2,
                          v29);
        if ( !v19 )
        {
          v56 = 257;
          v19 = sub_BD2C40(72, unk_3F10FD0);
          if ( v19 )
          {
            v31 = *(_QWORD ***)(a2 + 8);
            v32 = *((unsigned __int8 *)v31 + 8);
            if ( (unsigned int)(v32 - 17) > 1 )
            {
              v34 = sub_BCB2A0(*v31);
            }
            else
            {
              BYTE4(v48) = (_BYTE)v32 == 18;
              LODWORD(v48) = *((_DWORD *)v31 + 8);
              v33 = (__int64 *)sub_BCB2A0(*v31);
              v34 = sub_BCE1B0(v33, v48);
            }
            sub_B523C0((__int64)v19, v34, 53, v10, a2, v29, (__int64)&v54, 0, 0, 0);
          }
          (*(void (__fastcall **)(__int64, _QWORD *, const char **, __int64, __int64))(*(_QWORD *)v28[11] + 16LL))(
            v28[11],
            v19,
            &v50,
            v28[7],
            v28[8]);
          v35 = 16LL * *((unsigned int *)v28 + 2);
          v36 = *v28;
          v37 = v36 + v35;
          while ( v37 != v36 )
          {
            v38 = *(_QWORD *)(v36 + 8);
            v39 = *(_DWORD *)v36;
            v36 += 16;
            sub_B99FD0((__int64)v19, v39, v38);
          }
        }
        return v19;
      }
    }
    else if ( (*(_QWORD *)(v11 + 8LL * (v12 >> 6)) & (1LL << v12)) != 0 && (unsigned int)sub_C44590(a3) == v12 )
    {
      goto LABEL_28;
    }
  }
  else if ( v9 <= 0x40 )
  {
    if ( !*(_QWORD *)a3 )
      goto LABEL_23;
  }
  else if ( (unsigned int)sub_C444A0(a3) == v9 )
  {
    goto LABEL_23;
  }
  v13 = *(__int64 **)(a1 + 32);
  v44 = v13;
  v53 = 773;
  v50 = sub_BD5D20(a2);
  v51 = v14;
  v52 = ".off";
  v45 = sub_AD8D80(v47, a3);
  v15 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v13[10] + 32LL))(
          v13[10],
          15,
          a2,
          v45,
          0,
          0);
  if ( !v15 )
  {
    v56 = 257;
    v15 = sub_B504D0(15, a2, v45, (__int64)&v54, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v44[11] + 16LL))(
      v44[11],
      v15,
      &v50,
      v44[7],
      v44[8]);
    if ( *v44 != *v44 + 16LL * *((unsigned int *)v44 + 2) )
    {
      v46 = v10;
      v40 = *v44;
      v41 = *v44 + 16LL * *((unsigned int *)v44 + 2);
      do
      {
        v42 = *(_QWORD *)(v40 + 8);
        v43 = *(_DWORD *)v40;
        v40 += 16;
        sub_B99FD0(v15, v43, v42);
      }
      while ( v41 != v40 );
      v10 = v46;
    }
  }
  LODWORD(v51) = *(_DWORD *)(a4 + 8);
  if ( (unsigned int)v51 > 0x40 )
    sub_C43780((__int64)&v50, (const void **)a4);
  else
    v50 = *(const char **)a4;
  sub_C46B40((__int64)&v50, (__int64 *)a3);
  v16 = v51;
  LODWORD(v51) = 0;
  v55 = v16;
  v54 = v50;
  v17 = sub_AD8D80(v47, (__int64)&v54);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  if ( (unsigned int)v51 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
  v53 = 257;
  v18 = *(__int64 **)(a1 + 32);
  v19 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, __int64))(*(_QWORD *)v18[10] + 56LL))(
                    v18[10],
                    v10,
                    v15,
                    v17);
  if ( !v19 )
  {
    v56 = 257;
    v19 = sub_BD2C40(72, unk_3F10FD0);
    if ( v19 )
    {
      v20 = *(_QWORD ***)(v15 + 8);
      v21 = *((unsigned __int8 *)v20 + 8);
      if ( (unsigned int)(v21 - 17) > 1 )
      {
        v23 = sub_BCB2A0(*v20);
      }
      else
      {
        BYTE4(v49) = (_BYTE)v21 == 18;
        LODWORD(v49) = *((_DWORD *)v20 + 8);
        v22 = (__int64 *)sub_BCB2A0(*v20);
        v23 = sub_BCE1B0(v22, v49);
      }
      sub_B523C0((__int64)v19, v23, 53, v10, v15, v17, (__int64)&v54, 0, 0, 0);
    }
    (*(void (__fastcall **)(__int64, _QWORD *, const char **, __int64, __int64))(*(_QWORD *)v18[11] + 16LL))(
      v18[11],
      v19,
      &v50,
      v18[7],
      v18[8]);
    v24 = *v18;
    v25 = *v18 + 16LL * *((unsigned int *)v18 + 2);
    if ( *v18 != v25 )
    {
      do
      {
        v26 = *(_QWORD *)(v24 + 8);
        v27 = *(_DWORD *)v24;
        v24 += 16;
        sub_B99FD0((__int64)v19, v27, v26);
      }
      while ( v25 != v24 );
    }
  }
  return v19;
}
