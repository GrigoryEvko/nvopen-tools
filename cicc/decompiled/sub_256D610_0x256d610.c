// Function: sub_256D610
// Address: 0x256d610
//
__int64 __fastcall sub_256D610(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 (__fastcall *v8)(__int64); // rax
  __int64 v9; // rdi
  __int64 (__fastcall *v10)(__int64); // rax
  __int64 *v11; // rdx
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 *v17; // rax
  unsigned int *v18; // rax
  __int64 v19; // r14
  __int64 v20; // r12
  char v21; // bl
  __int64 *v22; // rax
  __int64 v23; // r8
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *v33; // rcx
  __int64 v34; // [rsp-10h] [rbp-130h]
  __int64 v35; // [rsp+8h] [rbp-118h]
  __int64 v36; // [rsp+10h] [rbp-110h]
  __int64 v38; // [rsp+20h] [rbp-100h]
  __int64 v39; // [rsp+28h] [rbp-F8h]
  __int64 v40; // [rsp+30h] [rbp-F0h]
  __int64 v41; // [rsp+38h] [rbp-E8h]
  __int64 v43; // [rsp+50h] [rbp-D0h]
  char v44; // [rsp+5Dh] [rbp-C3h]
  char v45; // [rsp+5Eh] [rbp-C2h]
  unsigned int v47; // [rsp+6Ch] [rbp-B4h] BYREF
  __int64 v48; // [rsp+70h] [rbp-B0h]
  __int64 v49; // [rsp+78h] [rbp-A8h]
  __int64 v50; // [rsp+80h] [rbp-A0h]
  __int64 v51; // [rsp+88h] [rbp-98h]
  _QWORD v52[2]; // [rsp+90h] [rbp-90h] BYREF
  __int64 *v53; // [rsp+A0h] [rbp-80h]
  __int64 *v54; // [rsp+A8h] [rbp-78h]
  __int64 *v55; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-68h]
  _QWORD v57[12]; // [rsp+C0h] [rbp-60h] BYREF

  v8 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 48LL);
  if ( v8 == sub_2534F10 )
    v9 = a3 + 88;
  else
    v9 = v8(a3);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v9 + 16LL))(v9) && (v44 = *(_BYTE *)(a1 + 393)) != 0 )
  {
    v47 = 1;
    v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a3 + 48LL);
    if ( v10 == sub_2534F10 )
      v38 = a3 + 88;
    else
      v38 = v10(a3);
    v11 = *(__int64 **)(v38 + 144);
    if ( *(_DWORD *)(v38 + 152) )
    {
      v54 = &v11[12 * *(unsigned int *)(v38 + 160)];
      v52[0] = v38 + 136;
      v13 = *(_QWORD *)(v38 + 136);
      v53 = v11;
      v52[1] = v13;
      sub_255DC40((__int64)v52);
      v35 = *(_QWORD *)(v38 + 144) + 96LL * *(unsigned int *)(v38 + 160);
      v17 = v53;
      if ( v53 != (__int64 *)v35 )
      {
        v43 = a1 + 88;
        if ( v53[11] )
          goto LABEL_44;
LABEL_13:
        v39 = v17[2];
        v36 = v39 + 4LL * *((unsigned int *)v17 + 6);
        v45 = v44;
LABEL_14:
        if ( v45 )
          goto LABEL_37;
LABEL_15:
        if ( v36 == v39 )
          goto LABEL_38;
        v18 = (unsigned int *)(v39 + 32);
        while ( 1 )
        {
          v19 = *(_QWORD *)(v38 + 8) + 112LL * *v18;
          if ( !a6 && *(_DWORD *)(*(_QWORD *)(v38 + 8) + 112LL * *v18 + 96) == 17 )
            goto LABEL_35;
          if ( *(_QWORD *)(a4 + 88) )
          {
            v21 = 0;
            v20 = *(_QWORD *)(a4 + 72);
            v41 = a4 + 56;
          }
          else
          {
            v20 = *(_QWORD *)a4;
            v21 = v44;
            v41 = *(_QWORD *)a4 + 8LL * *(unsigned int *)(a4 + 8);
          }
          if ( !v21 )
          {
            if ( v41 != v20 )
              goto LABEL_23;
            goto LABEL_35;
          }
          while ( 1 )
          {
            if ( v41 == v20 )
              goto LABEL_35;
            v22 = (__int64 *)v20;
LABEL_24:
            v23 = *v22;
            v55 = v57;
            if ( v23 == 0x7FFFFFFF )
            {
              v30 = 1;
              v56 = 0x300000001LL;
              v57[0] = 0x7FFFFFFF;
              v57[1] = 0x7FFFFFFF;
            }
            else
            {
              v56 = 0x300000000LL;
              if ( !*(_DWORD *)(v19 + 40) )
                goto LABEL_26;
              v40 = v23;
              sub_2538710((__int64)&v55, v19 + 32, v14, v15, v23, v16);
              v30 = (unsigned int)v56;
              v23 = v40;
              if ( !(_DWORD)v56 )
                goto LABEL_26;
            }
            v31 = v55;
            v32 = *v55;
            if ( *v55 != 0x7FFFFFFF && v55[1] != 0x7FFFFFFF )
            {
              v33 = &v55[2 * v30];
              while ( 1 )
              {
                v31 += 2;
                *(v31 - 2) = v23 + v32;
                if ( v31 == v33 )
                  break;
                v32 = *v31;
              }
            }
LABEL_26:
            v24 = *(_DWORD *)(v19 + 96);
            v25 = *(_QWORD *)(v19 + 16);
            v26 = *(_QWORD *)(v19 + 24);
            v50 = v25;
            v51 = v26;
            v48 = v25;
            if ( !a6 )
              v24 = v24 & 0xFFFFFFFC | 2;
            v27 = *(_QWORD *)(v19 + 104);
            v34 = *(_QWORD *)(v19 + 8);
            v49 = v26;
            v28 = sub_256C1D0(v43, a2, (__int64)&v55, a5, v25, v26, v24, v27, v34);
            sub_250C0C0((int *)&v47, v28);
            if ( v55 != v57 )
              _libc_free((unsigned __int64)v55);
            if ( !v21 )
              break;
            v20 += 8;
          }
          v20 = sub_220EF30(v20);
          if ( v41 != v20 )
          {
LABEL_23:
            v22 = (__int64 *)(v20 + 32);
            goto LABEL_24;
          }
LABEL_35:
          if ( !v45 )
          {
            v39 = sub_220EF30(v39);
            goto LABEL_15;
          }
          v39 += 4;
LABEL_37:
          v18 = (unsigned int *)v39;
          if ( v36 == v39 )
          {
LABEL_38:
            v17 = v53 + 12;
            v53 = v17;
            if ( v54 == v17 )
              goto LABEL_42;
            v15 = 0x7FFFFFFFFFFFFFFFLL;
            v29 = *v17;
            v14 = 0x7FFFFFFFFFFFFFFELL;
            if ( *v17 == 0x7FFFFFFFFFFFFFFFLL )
            {
LABEL_59:
              if ( v17[1] == 0x7FFFFFFFFFFFFFFFLL )
                goto LABEL_57;
            }
            else
            {
              while ( 2 )
              {
                if ( v29 == 0x7FFFFFFFFFFFFFFELL && v17[1] == 0x7FFFFFFFFFFFFFFELL )
                {
LABEL_57:
                  v17 += 12;
                  v53 = v17;
                  if ( v54 != v17 )
                  {
                    v29 = *v17;
                    if ( *v17 != 0x7FFFFFFFFFFFFFFFLL )
                      continue;
                    goto LABEL_59;
                  }
LABEL_42:
                  if ( (__int64 *)v35 == v17 )
                    return v47;
                  if ( v17[11] )
                  {
LABEL_44:
                    v45 = 0;
                    v36 = (__int64)(v17 + 7);
                    v39 = v17[9];
                    goto LABEL_14;
                  }
                  goto LABEL_13;
                }
                break;
              }
            }
            v17 = v53;
            goto LABEL_42;
          }
        }
      }
    }
    return v47;
  }
  else
  {
    *(_BYTE *)(a1 + 393) = *(_BYTE *)(a1 + 392);
    return 0;
  }
}
