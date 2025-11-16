// Function: sub_25A0CD0
// Address: 0x25a0cd0
//
__int64 __fastcall sub_25A0CD0(__int64 a1, unsigned __int64 a2)
{
  __int64 *v3; // r12
  __int64 v4; // r14
  __int64 v5; // rax
  _BYTE *v6; // r14
  __int64 (__fastcall *v7)(__int64); // rax
  unsigned __int8 *v8; // rdi
  __int64 (__fastcall *v9)(__int64); // rax
  unsigned int v10; // r13d
  __int64 (__fastcall *v11)(__int64); // rax
  __int64 (__fastcall *v13)(__int64); // rax
  __int64 v14; // rax
  __int64 *v15; // r14
  unsigned __int64 v16; // r14
  __int64 v17; // r12
  __int64 v18; // r14
  char (__fastcall *v19)(__int64, __int64, __int64, __int64, __int64); // r9
  __int64 *v20; // rax
  __int64 v21; // r8
  __int64 v22; // rdx
  int v23; // r10d
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // [rsp-10h] [rbp-90h]
  __int64 v29; // [rsp+0h] [rbp-80h]
  __int64 *v30; // [rsp+8h] [rbp-78h]
  __int64 *v31; // [rsp+18h] [rbp-68h]
  int v32; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v33; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int64 v34; // [rsp+38h] [rbp-48h]
  __int64 v35; // [rsp+40h] [rbp-40h]
  __int64 v36; // [rsp+48h] [rbp-38h]

  v3 = *(__int64 **)a1;
  v4 = **(_QWORD **)a1;
  sub_250D230((unsigned __int64 *)&v33, a2, 5, 0);
  v5 = sub_251C7D0(v4, v33, v34, v3[1], 1, 0, 1);
  if ( v5 )
  {
    v6 = (_BYTE *)v5;
    v7 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v5 + 48LL);
    v8 = v7 == sub_2534F50 ? v6 + 88 : (unsigned __int8 *)((__int64 (__fastcall *)(_BYTE *, __int64))v7)(v6, v28);
    v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL);
    v10 = v9 == sub_2505E30 ? v8[9] : ((__int64 (__fastcall *)(unsigned __int8 *, __int64))v9)(v8, v28);
    if ( (_BYTE)v10 )
    {
      v11 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 120LL);
      if ( !(v11 == sub_2534E40 ? v6[168] : ((__int64 (__fastcall *)(_BYTE *, __int64))v11)(v6, v28)) )
      {
        v13 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v6 + 112LL);
        if ( v13 == sub_2534E30 )
          v14 = (__int64)(v6 + 120);
        else
          v14 = ((__int64 (__fastcall *)(_BYTE *, __int64))v13)(v6, v28);
        v15 = *(__int64 **)(v14 + 32);
        v30 = &v15[*(unsigned int *)(v14 + 40)];
        if ( v15 == v30 )
          return v10;
        v31 = *(__int64 **)(v14 + 32);
        while ( 1 )
        {
          v16 = *v31;
          if ( *v31 == *(_QWORD *)(v3[2] + 8) )
            break;
          if ( sub_B2FC80(*v31) )
          {
            if ( !(unsigned __int8)sub_B2D610(v16, 24) )
              break;
          }
          else if ( v16 == sub_25096F0((_QWORD *)(v3[1] + 72)) )
          {
            if ( *(_QWORD *)v3[3] != *(_QWORD *)v3[2] )
              break;
          }
          else
          {
            v29 = *v3;
            sub_250D230((unsigned __int64 *)&v33, v16, 4, 0);
            v25 = sub_25289A0(v29, v33, v34, v3[1], 1, 0, 1);
            v26 = *(_QWORD *)(v16 + 80);
            if ( !v26 )
              BUG();
            v27 = *(_QWORD *)(v26 + 32);
            if ( v27 )
              v27 -= 24;
            if ( !v25
              || (*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v25 + 112LL))(
                   v25,
                   *v3,
                   v27,
                   *(_QWORD *)(v3[2] + 8),
                   *(_QWORD *)(v3[2] + 16)) )
            {
              break;
            }
          }
          if ( v30 == ++v31 )
            return v10;
        }
      }
    }
  }
  v10 = 0;
  v17 = **(_QWORD **)(a1 + 8);
  if ( v17 )
  {
    v18 = *(_QWORD *)(a1 + 16);
    v19 = *(char (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v17 + 112LL);
    v20 = *(__int64 **)(a1 + 24);
    v21 = v20[2];
    v22 = *v20;
    if ( v19 != sub_257FCC0 )
    {
      v23 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64, unsigned __int64, __int64))v19)(
              v17,
              *(_QWORD *)(a1 + 16),
              v22,
              a2,
              v21);
      return v23 ^ 1u;
    }
    if ( a2 != v22 )
    {
      v33 = *v20;
      v34 = a2;
      v35 = v21;
      v36 = 0;
      if ( !v21 || *(_DWORD *)(v21 + 20) == *(_DWORD *)(v21 + 24) )
        v35 = 0;
      if ( (unsigned __int8)sub_2573570(v17, &v33, &v32) )
        LOBYTE(v23) = v32 == 1;
      else
        v23 = sub_257EF80(v17, v18, (unsigned __int64)&v33, 1);
      return v23 ^ 1u;
    }
  }
  return v10;
}
