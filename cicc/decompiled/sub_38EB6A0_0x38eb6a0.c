// Function: sub_38EB6A0
// Address: 0x38eb6a0
//
char __fastcall sub_38EB6A0(__int64 a1, __int64 *a2, __int64 a3)
{
  _QWORD **v6; // rdi
  __int64 (*v7)(void); // rax
  char v8; // al
  char result; // al
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int64 v16; // rdi
  __int64 v17; // rsi
  unsigned __int64 v18; // rsi
  __int64 v19; // rdi
  unsigned __int16 v20; // ax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rcx
  unsigned __int64 v26; // rcx
  unsigned __int64 v27; // rax
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // [rsp+0h] [rbp-90h] BYREF
  unsigned __int64 v36; // [rsp+8h] [rbp-88h]
  const char *v37; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 *v38; // [rsp+18h] [rbp-78h]
  __int64 v39; // [rsp+20h] [rbp-70h]
  char *v40; // [rsp+30h] [rbp-60h]
  char v41; // [rsp+40h] [rbp-50h]
  char v42; // [rsp+41h] [rbp-4Fh]
  __int64 v43[2]; // [rsp+50h] [rbp-40h] BYREF
  __int64 v44; // [rsp+60h] [rbp-30h]

  *a2 = 0;
  v6 = *(_QWORD ***)(a1 + 8);
  v7 = (__int64 (*)(void))(*v6)[3];
  if ( (char *)v7 == (char *)sub_38E29C0 )
    v8 = (*(__int64 (__fastcall **)(_QWORD *))(*v6[1] + 184LL))(v6[1]);
  else
    v8 = v7();
  if ( v8 || (unsigned __int8)sub_38EB510(a1, 1u, a2, a3) )
    return 1;
  if ( **(_DWORD **)(a1 + 152) == 45 )
  {
    sub_38EB180(a1);
    if ( **(_DWORD **)(a1 + 152) == 2 )
    {
      v12 = sub_3909460(a1);
      if ( *(_DWORD *)v12 == 2 )
      {
        v19 = *(_QWORD *)(v12 + 8);
        v17 = *(_QWORD *)(v12 + 16);
      }
      else
      {
        v16 = *(_QWORD *)(v12 + 16);
        v17 = 0;
        if ( v16 )
        {
          v13 = 1;
          v18 = v16 - 1;
          if ( v16 == 1 )
            v18 = 1;
          if ( v18 > v16 )
            v18 = *(_QWORD *)(v12 + 16);
          v16 = 1;
          v17 = v18 - 1;
        }
        v19 = *(_QWORD *)(v12 + 8) + v16;
      }
      v20 = sub_38CBC00(v19, v17, v13, v14, v15);
      if ( v20 == 1 )
      {
        v42 = 1;
        v40 = "'";
        v41 = 3;
        v22 = sub_3909460(a1);
        v23 = v22;
        if ( *(_DWORD *)v22 == 2 )
        {
          v27 = *(_QWORD *)(v22 + 8);
          v25 = *(_QWORD *)(v23 + 16);
        }
        else
        {
          v24 = *(_QWORD *)(v22 + 16);
          v25 = 0;
          if ( v24 )
          {
            v26 = v24 - 1;
            if ( v24 == 1 )
              v26 = 1;
            if ( v26 > v24 )
              v26 = v24;
            v24 = 1;
            v25 = v26 - 1;
          }
          v27 = *(_QWORD *)(v23 + 8) + v24;
        }
        v35 = v27;
        v37 = "invalid variant '";
        v38 = &v35;
        v28 = v41;
        v36 = v25;
        LOWORD(v39) = 1283;
      }
      else
      {
        v21 = sub_38E57D0(a1, *a2, v20);
        if ( v21 )
        {
          *a2 = v21;
          sub_38EB180(a1);
          goto LABEL_8;
        }
        v42 = 1;
        v40 = "' (no symbols present)";
        v41 = 3;
        v29 = sub_3909460(a1);
        v30 = v29;
        if ( *(_DWORD *)v29 == 2 )
        {
          v34 = *(_QWORD *)(v29 + 8);
          v32 = *(_QWORD *)(v30 + 16);
        }
        else
        {
          v31 = *(_QWORD *)(v29 + 16);
          v32 = 0;
          if ( v31 )
          {
            v33 = v31 - 1;
            if ( v31 == 1 )
              v33 = 1;
            if ( v33 > v31 )
              v33 = v31;
            v31 = 1;
            v32 = v33 - 1;
          }
          v34 = *(_QWORD *)(v30 + 8) + v31;
        }
        v35 = v34;
        v37 = "invalid modifier '";
        v38 = &v35;
        v28 = v41;
        v36 = v32;
        LOWORD(v39) = 1283;
      }
      v43[1] = (__int64)v40;
      v43[0] = (__int64)&v37;
      LOBYTE(v44) = 2;
      BYTE1(v44) = v28;
    }
    else
    {
      v43[0] = (__int64)"unexpected symbol modifier following '@'";
      LOWORD(v44) = 259;
    }
    return sub_3909CF0(a1, v43, 0, 0, v10, v11);
  }
LABEL_8:
  result = sub_38CF290(*a2, v43);
  if ( result )
  {
    *a2 = sub_38CB470(v43[0], *(_QWORD *)(a1 + 320));
    return 0;
  }
  return result;
}
