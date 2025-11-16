// Function: sub_1F34980
// Address: 0x1f34980
//
__int64 __fastcall sub_1F34980(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 *v3; // r8
  __int64 *v4; // r9
  __int64 *v5; // rbx
  __int64 *v6; // r13
  __int64 v7; // rsi
  __int64 *v8; // rdi
  __int64 *v9; // rax
  __int64 *v10; // rcx
  _BYTE *v11; // r14
  _BYTE *v12; // r12
  __int64 v13; // rbx
  __int64 *v14; // rdi
  __int64 *v15; // r8
  int v16; // eax
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 *v19; // rax
  unsigned __int64 v20; // rsi
  _QWORD *v21; // r12
  __int64 *v22; // r12
  __int64 v23; // r13
  __int64 v24; // rdi
  __int64 (*v25)(); // rax
  __int64 v26; // r8
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // rcx
  int v30; // r8d
  int v31; // r9d
  __int64 v32; // rax
  __int64 v33; // rsi
  __int64 *v35; // rdx
  unsigned __int8 v36; // [rsp+17h] [rbp-1F9h]
  _QWORD *v38; // [rsp+20h] [rbp-1F0h]
  __int64 *v41; // [rsp+40h] [rbp-1D0h]
  __int64 *v42; // [rsp+48h] [rbp-1C8h]
  __int64 v43; // [rsp+48h] [rbp-1C8h]
  _QWORD *v44; // [rsp+58h] [rbp-1B8h] BYREF
  _QWORD *v45; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v46; // [rsp+68h] [rbp-1A8h] BYREF
  __int64 *v47; // [rsp+70h] [rbp-1A0h] BYREF
  __int64 v48; // [rsp+78h] [rbp-198h]
  _BYTE dest[64]; // [rsp+80h] [rbp-190h] BYREF
  __int64 v50; // [rsp+C0h] [rbp-150h] BYREF
  __int64 *v51; // [rsp+C8h] [rbp-148h]
  __int64 *v52; // [rsp+D0h] [rbp-140h]
  __int64 v53; // [rsp+D8h] [rbp-138h]
  int i; // [rsp+E0h] [rbp-130h]
  _BYTE v55[72]; // [rsp+E8h] [rbp-128h] BYREF
  _BYTE *v56; // [rsp+130h] [rbp-E0h] BYREF
  __int64 v57; // [rsp+138h] [rbp-D8h]
  _BYTE v58[208]; // [rsp+140h] [rbp-D0h] BYREF

  v3 = (__int64 *)v55;
  v4 = (__int64 *)v55;
  v5 = (__int64 *)a2[12];
  v6 = (__int64 *)a2[11];
  v50 = 0;
  v51 = (__int64 *)v55;
  v52 = (__int64 *)v55;
  v53 = 8;
  for ( i = 0; v5 != v6; ++v6 )
  {
LABEL_5:
    v7 = *v6;
    if ( v3 != v4 )
      goto LABEL_3;
    v8 = &v3[HIDWORD(v53)];
    if ( v8 != v3 )
    {
      v9 = v3;
      v10 = 0;
      while ( v7 != *v9 )
      {
        if ( *v9 == -2 )
          v10 = v9;
        if ( v8 == ++v9 )
        {
          if ( !v10 )
            goto LABEL_81;
          ++v6;
          *v10 = v7;
          v4 = v52;
          --i;
          v3 = v51;
          ++v50;
          if ( v5 != v6 )
            goto LABEL_5;
          goto LABEL_14;
        }
      }
      continue;
    }
LABEL_81:
    if ( HIDWORD(v53) < (unsigned int)v53 )
    {
      ++HIDWORD(v53);
      *v8 = v7;
      v3 = v51;
      ++v50;
      v4 = v52;
    }
    else
    {
LABEL_3:
      sub_16CCBA0((__int64)&v50, v7);
      v4 = v52;
      v3 = v51;
    }
  }
LABEL_14:
  v48 = 0x800000000LL;
  v11 = (_BYTE *)a2[9];
  v12 = (_BYTE *)a2[8];
  v47 = (__int64 *)dest;
  v13 = (v11 - v12) >> 3;
  if ( (unsigned __int64)(v11 - v12) > 0x40 )
  {
    sub_16CD150((__int64)&v47, dest, (v11 - v12) >> 3, 8, (int)v3, (int)v4);
    v14 = v47;
    v16 = v48;
    v15 = &v47[(unsigned int)v48];
  }
  else
  {
    v14 = (__int64 *)dest;
    v15 = (__int64 *)dest;
    v16 = 0;
  }
  if ( v12 != v11 )
  {
    memmove(v15, v12, v11 - v12);
    v14 = v47;
    v16 = v48;
  }
  LODWORD(v48) = v13 + v16;
  v41 = &v14[(unsigned int)(v13 + v16)];
  if ( v41 != v14 )
  {
    v36 = 0;
    v17 = v14;
LABEL_28:
    while ( 1 )
    {
      v21 = (_QWORD *)*v17;
      if ( !(unsigned __int8)sub_1DD61A0(*v17) )
        break;
LABEL_27:
      if ( v41 == ++v17 )
        goto LABEL_66;
    }
    v42 = (__int64 *)v21[12];
    if ( v42 == (__int64 *)v21[11] )
      goto LABEL_43;
    v20 = (unsigned __int64)v52;
    v38 = v21;
    v22 = (__int64 *)v21[11];
    while ( 1 )
    {
      v19 = v51;
      v23 = *v22;
      if ( (__int64 *)v20 == v51 )
      {
        v18 = (__int64 *)(v20 + 8LL * HIDWORD(v53));
        if ( (__int64 *)v20 == v18 )
        {
          v35 = (__int64 *)v20;
        }
        else
        {
          do
          {
            if ( v23 == *v19 )
              break;
            ++v19;
          }
          while ( v18 != v19 );
          v35 = (__int64 *)(v20 + 8LL * HIDWORD(v53));
        }
LABEL_39:
        while ( v35 != v19 )
        {
          if ( (unsigned __int64)*v19 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_23;
          ++v19;
        }
        if ( v18 != v19 )
          goto LABEL_24;
      }
      else
      {
        v18 = (__int64 *)(v20 + 8LL * (unsigned int)v53);
        v19 = sub_16CC9F0((__int64)&v50, *v22);
        if ( v23 == *v19 )
        {
          v20 = (unsigned __int64)v52;
          if ( v52 == v51 )
            v35 = &v52[HIDWORD(v53)];
          else
            v35 = &v52[(unsigned int)v53];
          goto LABEL_39;
        }
        v20 = (unsigned __int64)v52;
        if ( v52 == v51 )
        {
          v19 = &v52[HIDWORD(v53)];
          v35 = v19;
          goto LABEL_39;
        }
        v19 = &v52[(unsigned int)v53];
LABEL_23:
        if ( v18 != v19 )
        {
LABEL_24:
          if ( v23 + 24 != (*(_QWORD *)(v23 + 24) & 0xFFFFFFFFFFFFFFF8LL)
            && (**(_WORD **)(*(_QWORD *)(v23 + 32) + 16LL) == 45 || !**(_WORD **)(*(_QWORD *)(v23 + 32) + 16LL)) )
          {
            goto LABEL_27;
          }
        }
      }
      if ( v42 == ++v22 )
      {
        v21 = v38;
LABEL_43:
        v44 = 0;
        v57 = 0x400000000LL;
        v45 = 0;
        v24 = *a1;
        v56 = v58;
        v25 = *(__int64 (**)())(*(_QWORD *)v24 + 264LL);
        if ( v25 == sub_1D820E0 )
          goto LABEL_27;
        if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD *, _QWORD **, _QWORD **, _BYTE **, _QWORD))v25)(
               v24,
               v21,
               &v44,
               &v45,
               &v56,
               0) )
        {
          if ( v56 != v58 )
            _libc_free((unsigned __int64)v56);
          goto LABEL_27;
        }
        v26 = *(_QWORD *)a2[11];
        v27 = (_QWORD *)v21[1];
        if ( v27 == (_QWORD *)(v21[7] + 320LL) )
          v27 = 0;
        v28 = v44;
        if ( !(_DWORD)v57 )
        {
          v45 = v44;
          v29 = v44;
          if ( !v44 )
          {
LABEL_79:
            v44 = v27;
            v28 = v27;
            if ( !v29 )
              goto LABEL_83;
          }
LABEL_50:
          if ( a2 == v29 )
            goto LABEL_84;
LABEL_51:
          if ( a2 == v28 )
            goto LABEL_85;
LABEL_52:
          if ( v28 == v29 )
            goto LABEL_86;
          goto LABEL_53;
        }
        v29 = v45;
        if ( !v44 )
          goto LABEL_79;
        if ( v45 )
          goto LABEL_50;
LABEL_83:
        v45 = v27;
        v29 = v27;
        if ( a2 != v27 )
          goto LABEL_51;
LABEL_84:
        v45 = (_QWORD *)v26;
        v29 = (_QWORD *)v26;
        if ( a2 != v28 )
          goto LABEL_52;
LABEL_85:
        v28 = (_QWORD *)v26;
        v44 = (_QWORD *)v26;
        if ( (_QWORD *)v26 == v29 )
        {
LABEL_86:
          LODWORD(v57) = 0;
          v45 = 0;
          if ( v27 )
          {
            if ( v28 == v27 )
              goto LABEL_88;
          }
          else if ( !v28 )
          {
            v44 = 0;
          }
          goto LABEL_55;
        }
LABEL_53:
        if ( v29 == v27 )
        {
          v45 = 0;
        }
        else if ( v28 == v27 && !v29 )
        {
LABEL_88:
          v44 = 0;
        }
LABEL_55:
        v43 = v26;
        sub_1DD6F40(&v46, (__int64)v21);
        (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD))(*(_QWORD *)*a1 + 280LL))(*a1, v21, 0);
        if ( sub_1DD6970((__int64)v21, v43) )
          sub_1DD91B0((__int64)v21, (__int64)a2);
        else
          sub_1DD9570((__int64)v21, (__int64)a2, v43);
        if ( v44 )
          (*(void (__fastcall **)(_QWORD, _QWORD *, _QWORD *, _QWORD *, _BYTE *, _QWORD, __int64 *, _QWORD))(*(_QWORD *)*a1 + 288LL))(
            *a1,
            v21,
            v44,
            v45,
            v56,
            (unsigned int)v57,
            &v46,
            0);
        v32 = *(unsigned int *)(a3 + 8);
        if ( (unsigned int)v32 >= *(_DWORD *)(a3 + 12) )
        {
          sub_16CD150(a3, (const void *)(a3 + 16), 0, 8, v30, v31);
          v32 = *(unsigned int *)(a3 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a3 + 8 * v32) = v21;
        v33 = v46;
        ++*(_DWORD *)(a3 + 8);
        if ( v33 )
          sub_161E7C0((__int64)&v46, v33);
        if ( v56 != v58 )
          _libc_free((unsigned __int64)v56);
        v36 = 1;
        if ( v41 == ++v17 )
        {
LABEL_66:
          v14 = v47;
          goto LABEL_67;
        }
        goto LABEL_28;
      }
    }
  }
  v36 = 0;
LABEL_67:
  if ( v14 != (__int64 *)dest )
    _libc_free((unsigned __int64)v14);
  if ( v52 != v51 )
    _libc_free((unsigned __int64)v52);
  return v36;
}
