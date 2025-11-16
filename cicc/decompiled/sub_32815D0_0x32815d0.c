// Function: sub_32815D0
// Address: 0x32815d0
//
__int64 __fastcall sub_32815D0(__int64 a1, __int64 a2, char a3)
{
  unsigned int *v4; // rdx
  _QWORD *v5; // rcx
  __int64 v7; // rax
  __int64 v9; // rdx
  int v10; // esi
  _QWORD *v11; // rax
  __int64 v12; // r12
  __int64 v13; // r8
  __int64 v14; // r13
  unsigned int v15; // edx
  __int64 v16; // rcx
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int16 *v19; // rax
  bool v20; // zf
  __int16 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rax
  __int16 v24; // r14
  __int64 v25; // rdx
  int v26; // eax
  __int64 v27; // rdi
  __int64 (*v28)(); // rax
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // r13
  __int64 v32; // r12
  __int128 v33; // rax
  int v34; // r9d
  __int128 v35; // [rsp-98h] [rbp-98h]
  __int64 v36; // [rsp-78h] [rbp-78h]
  __int64 v37; // [rsp-70h] [rbp-70h]
  unsigned int v38; // [rsp-70h] [rbp-70h]
  unsigned int v39; // [rsp-68h] [rbp-68h] BYREF
  __int64 v40; // [rsp-60h] [rbp-60h]
  __int64 v41; // [rsp-58h] [rbp-58h] BYREF
  int v42; // [rsp-50h] [rbp-50h]
  _QWORD v43[8]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v44; // [rsp-8h] [rbp-8h] BYREF

  if ( a3 )
    return 0;
  v4 = *(unsigned int **)(a1 + 40);
  v5 = *(_QWORD **)v4;
  if ( *(_DWORD *)(*(_QWORD *)v4 + 24LL) == 208 )
  {
    v7 = v5[7];
    if ( v7 )
    {
      v9 = v4[2];
      v10 = 1;
      do
      {
        if ( *(_DWORD *)(v7 + 8) == (_DWORD)v9 )
        {
          if ( !v10 )
            return 0;
          v7 = *(_QWORD *)(v7 + 32);
          if ( !v7 )
            goto LABEL_15;
          if ( *(_DWORD *)(v7 + 8) == (_DWORD)v9 )
            return 0;
          v10 = 0;
        }
        v7 = *(_QWORD *)(v7 + 32);
      }
      while ( v7 );
      if ( v10 == 1 )
        return 0;
LABEL_15:
      if ( *(_WORD *)(v5[6] + 16 * v9) == 2 )
      {
        v11 = (_QWORD *)v5[5];
        v12 = *v11;
        v13 = *v11;
        v14 = v11[1];
        v15 = *((_DWORD *)v11 + 2);
        v16 = v11[10];
        v17 = v11[5];
        v18 = v11[6];
        v19 = *(__int16 **)(a1 + 48);
        v20 = *(_DWORD *)(v16 + 96) == 18;
        v21 = *v19;
        v22 = *((_QWORD *)v19 + 1);
        LOWORD(v39) = v21;
        v40 = v22;
        if ( v20 )
        {
          v23 = *(_QWORD *)(v13 + 48) + 16LL * v15;
          v24 = *(_WORD *)v23;
          v37 = *(_QWORD *)(v23 + 8);
          if ( (unsigned __int8)sub_33CF460(v17, v18) )
          {
            if ( v24 == (_WORD)v39 && (v24 || v37 == v40) )
            {
              v41 = *(_QWORD *)(a1 + 80);
              if ( v41 )
                sub_325F5D0(&v41);
              v42 = *(_DWORD *)(a1 + 72);
              v43[0] = sub_2D5B750((unsigned __int16 *)&v39);
              v43[1] = v25;
              v26 = sub_CA1930(v43);
              v27 = *(_QWORD *)(a2 + 16);
              v38 = v26 - 1;
              v28 = *(__int64 (**)())(*(_QWORD *)v27 + 1728LL);
              if ( v28 == sub_2FE3600
                || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD))v28)(v27, v39, v40, v38) )
              {
                v29 = sub_34074A0(a2, &v41, v12, v14, v39, v40);
                v31 = v30;
                v32 = v29;
                *(_QWORD *)&v33 = sub_3400BD0(a2, v38, (unsigned int)&v44 - 80, v39, v40, 0, 0);
                *((_QWORD *)&v35 + 1) = v31;
                *(_QWORD *)&v35 = v32;
                v36 = sub_3406EB0(
                        a2,
                        (unsigned int)(*(_DWORD *)(a1 + 24) != 213) + 191,
                        (unsigned int)&v44 - 80,
                        v39,
                        v40,
                        v34,
                        v35,
                        v33);
                sub_9C6650(&v41);
                return v36;
              }
              sub_9C6650(&v41);
            }
          }
        }
      }
    }
  }
  return 0;
}
