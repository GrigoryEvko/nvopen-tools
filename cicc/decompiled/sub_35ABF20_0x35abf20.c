// Function: sub_35ABF20
// Address: 0x35abf20
//
__int64 __fastcall sub_35ABF20(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v8; // rbx
  __int64 *v9; // rdi
  __int64 v10; // rax
  __int64 (*v11)(); // rdx
  __int64 v12; // r14
  __int16 v13; // ax
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rdx
  _QWORD *v19; // r13
  __int64 v20; // rbx
  __int64 v21; // rbx
  __int64 v22; // rax
  unsigned int v23; // r8d
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 v28; // r14
  __int64 v29; // rdx
  __int64 (__fastcall *v30)(__int64); // rax
  __int64 v31; // rax
  _BYTE *v32; // rax
  int v33; // r15d
  __int64 v34; // [rsp-10h] [rbp-A0h]
  __int64 v35; // [rsp-8h] [rbp-98h]
  __int64 v36; // [rsp+8h] [rbp-88h]
  int v37; // [rsp+1Ch] [rbp-74h] BYREF
  __int64 v38[2]; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v39; // [rsp+30h] [rbp-60h] BYREF
  __int64 v40; // [rsp+38h] [rbp-58h]
  _QWORD v41[10]; // [rsp+40h] [rbp-50h] BYREF

  v5 = 0;
  v6 = (int)a4;
  v8 = a3;
  v9 = *(__int64 **)(a1 + 16);
  v10 = *v9;
  v11 = *(__int64 (**)())(*v9 + 136);
  if ( v11 != sub_2DD19D0 )
  {
    v5 = ((__int64 (__fastcall *)(__int64 *, __int64, __int64 (*)(), __int64, __int64, _QWORD))v11)(
           v9,
           a2,
           v11,
           a4,
           a5,
           0);
    v10 = **(_QWORD **)(a1 + 16);
  }
  v36 = v5;
  v12 = (*(__int64 (**)(void))(v10 + 200))();
  v13 = *(_WORD *)(a2 + 68);
  if ( (unsigned __int16)(v13 - 14) <= 1u )
  {
    v14 = *(_QWORD *)(a2 + 32);
    v37 = 0;
    v15 = v14 + 40 * v8;
    v16 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL)
                    + 40LL * (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 48) + 32LL) + *(_DWORD *)(v15 + 24))
                    + 8);
    v17 = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, int *))(*(_QWORD *)v36 + 224LL))(
            v36,
            a1,
            *(unsigned int *)(v15 + 24),
            &v37);
    v38[1] = v18;
    v38[0] = v17;
    sub_2EAB560((char *)v15, v37, 0, 0, 0, 0, 0, 0);
    v19 = (_QWORD *)sub_2E891C0(a2);
    if ( *(_WORD *)(a2 + 68) != 14 )
    {
      v20 = (v15 - *(_QWORD *)(a2 + 32) - 80) >> 3;
      v40 = 0x300000000LL;
      v39 = v41;
      (*(void (__fastcall **)(__int64, __int64 *, _BYTE **, unsigned __int64, __int64, __int64))(*(_QWORD *)v12 + 592LL))(
        v12,
        v38,
        &v39,
        0xCCCCCCCCCCCCCCCDLL,
        v34,
        v35);
      v21 = sub_B0DBA0(v19, v39, (unsigned int)v40, -858993459 * (int)v20, 0);
      if ( v39 != (_BYTE *)v41 )
        _libc_free((unsigned __int64)v39);
      goto LABEL_7;
    }
    v32 = *(_BYTE **)(a2 + 32);
    if ( v32[40] != 1 || *v32 )
    {
      v33 = (unsigned __int8)(4 * ((unsigned __int8)sub_AF4500((__int64)v19) == 0));
      if ( *(_WORD *)(a2 + 68) != 14 || (v32 = *(_BYTE **)(a2 + 32), v32[40] != 1) )
      {
LABEL_16:
        v21 = sub_2FF7570(v12, v19, v33, v38);
LABEL_7:
        v22 = sub_2E891A0(a2);
        v23 = 1;
        *(_QWORD *)(v22 + 24) = v21;
        return v23;
      }
    }
    else
    {
      LOBYTE(v33) = 0;
    }
    if ( !*v32 )
    {
      if ( (unsigned __int8)sub_AF4460((__int64)v19) )
      {
        v41[1] = (unsigned int)v16;
        v39 = v41;
        v41[0] = 148;
        v40 = 0x200000002LL;
        v19 = (_QWORD *)sub_B0D8A0(v19, (__int64)&v39, 1, 0);
        sub_2EAB560((char *)(*(_QWORD *)(a2 + 32) + 40LL), 0, 0, 0, 0, 0, 0, 0);
        if ( v39 != (_BYTE *)v41 )
          _libc_free((unsigned __int64)v39);
      }
    }
    goto LABEL_16;
  }
  v23 = 1;
  if ( v13 != 17 )
  {
    v23 = 0;
    if ( v13 == 32 )
    {
      LODWORD(v39) = 0;
      v25 = *(_QWORD *)(a2 + 32);
      v26 = (unsigned int)(v8 + 1);
      v27 = 40 * v8;
      v28 = v25 + 40 * v26;
      v29 = *(unsigned int *)(v25 + v27 + 24);
      v30 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v36 + 232LL);
      if ( v30 == sub_2FDBC50 )
        v31 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _BYTE **))(*(_QWORD *)v36 + 224LL))(
                v36,
                a1,
                v29,
                &v39);
      else
        v31 = ((__int64 (__fastcall *)(__int64, __int64, __int64, _BYTE **, _QWORD))v30)(v36, a1, v29, &v39, 0);
      *(_QWORD *)(v28 + 24) += v6 + v31;
      sub_2EAB560((char *)(*(_QWORD *)(a2 + 32) + v27), (int)v39, 0, 0, 0, 0, 0, 0);
      return 1;
    }
  }
  return v23;
}
