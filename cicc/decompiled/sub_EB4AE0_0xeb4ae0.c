// Function: sub_EB4AE0
// Address: 0xeb4ae0
//
__int64 __fastcall sub_EB4AE0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 (*v11)(); // rax
  char v12; // al
  unsigned int v13; // r8d
  char v15; // al
  char v16; // al
  __int64 v17; // r13
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  _QWORD *v21; // rsi
  __int64 v23; // rax
  __int64 v24; // [rsp+0h] [rbp-1C0h]
  unsigned __int8 v27; // [rsp+10h] [rbp-1B0h]
  unsigned __int8 v29; // [rsp+18h] [rbp-1A8h]
  __int64 v30; // [rsp+20h] [rbp-1A0h] BYREF
  unsigned __int64 v31; // [rsp+28h] [rbp-198h] BYREF
  _QWORD v32[2]; // [rsp+30h] [rbp-190h] BYREF
  __int64 v33; // [rsp+40h] [rbp-180h]
  __int64 v34; // [rsp+48h] [rbp-178h]
  __int64 v35; // [rsp+50h] [rbp-170h]
  __int64 v36; // [rsp+58h] [rbp-168h]
  _QWORD *v37; // [rsp+60h] [rbp-160h]
  _QWORD v38[2]; // [rsp+70h] [rbp-150h] BYREF
  __int64 v39; // [rsp+80h] [rbp-140h]
  char v40; // [rsp+88h] [rbp-138h] BYREF
  __int16 v41; // [rsp+90h] [rbp-130h]

  v5 = sub_ECD7B0(a1);
  v6 = sub_ECD6A0(v5);
  v38[0] = 0;
  v7 = v6;
  if ( sub_EAC4D0(a1, &v30, (__int64)v38) )
    return 1;
  v8 = *(_QWORD *)(a1 + 232);
  v9 = 0;
  v10 = v30;
  v11 = *(__int64 (**)())(*(_QWORD *)v8 + 80LL);
  if ( v11 != sub_C13ED0 )
  {
    v24 = v30;
    v23 = ((__int64 (__fastcall *)(__int64, __int64 *, _QWORD))v11)(v8, &v30, 0);
    v10 = v24;
    v9 = v23;
  }
  v12 = sub_E81930(v10, &v31, v9);
  if ( !v12 )
  {
    v32[0] = "unexpected token in '";
    LOWORD(v35) = 1283;
    v33 = a3;
    v41 = 770;
    v34 = a4;
    v38[0] = v32;
    v39 = (__int64)"' directive";
    return (unsigned int)sub_ECDA70(a1, v7, v38, 0, 0);
  }
  v29 = v12;
  v38[0] = "Count is negative";
  v41 = 259;
  v15 = sub_ECE070(a1, v31 >> 63, v7, v38);
  v13 = v29;
  if ( !v15 )
  {
    v16 = sub_ECE000(a1);
    v13 = v29;
    if ( !v16 )
    {
      v17 = sub_EB4420(a1, a2);
      if ( v17 )
      {
        v38[0] = &v40;
        v36 = 0x100000000LL;
        v37 = v38;
        v38[1] = 0;
        v39 = 256;
        v32[1] = 2;
        v33 = 0;
        v34 = 0;
        v35 = 0;
        v32[0] = &unk_49DD288;
        sub_CB5980((__int64)v32, 0, 0, 0);
        while ( v31-- )
        {
          v21 = v32;
          v19 = (unsigned int)sub_EA4200(a1, (int *)v32, v17, 0, 0, 0, 0, 0);
          if ( (_BYTE)v19 )
            goto LABEL_14;
        }
        v21 = (_QWORD *)a2;
        sub_EB41F0(a1, a2, v32, v18, v19, v20);
        LOBYTE(v19) = 0;
LABEL_14:
        v27 = v19;
        v32[0] = &unk_49DD388;
        sub_CB5840((__int64)v32);
        v13 = v27;
        if ( (char *)v38[0] != &v40 )
        {
          _libc_free(v38[0], v21);
          return v27;
        }
        return v13;
      }
      return 1;
    }
  }
  return v13;
}
