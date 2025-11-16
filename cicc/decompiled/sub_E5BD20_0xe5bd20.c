// Function: sub_E5BD20
// Address: 0xe5bd20
//
__int64 __fastcall sub_E5BD20(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rbp
  __int64 v4; // r12
  __int64 result; // rax
  int v7; // eax
  __int64 v8; // rdi
  unsigned int v9; // edx
  unsigned int v10; // r13d
  __int64 v11; // rax
  __int64 (*v12)(); // rax
  __int64 v13; // rdi
  __int64 v14; // rdi
  const char *v15; // rax
  __int64 v16; // rsi
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 *v22; // rdi
  __int64 (*v23)(); // rcx
  __int64 v24; // rdi
  __int64 v25; // rdi
  const char *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // [rsp-128h] [rbp-128h] BYREF
  __int64 v29; // [rsp-120h] [rbp-120h] BYREF
  _QWORD v30[2]; // [rsp-118h] [rbp-118h] BYREF
  __int64 v31; // [rsp-108h] [rbp-108h]
  int v32; // [rsp-100h] [rbp-100h]
  _QWORD v33[4]; // [rsp-F8h] [rbp-F8h] BYREF
  __int16 v34; // [rsp-D8h] [rbp-D8h]
  _QWORD v35[4]; // [rsp-C8h] [rbp-C8h] BYREF
  __int16 v36; // [rsp-A8h] [rbp-A8h]
  _QWORD v37[4]; // [rsp-98h] [rbp-98h] BYREF
  __int16 v38; // [rsp-78h] [rbp-78h]
  _QWORD v39[4]; // [rsp-68h] [rbp-68h] BYREF
  __int16 v40; // [rsp-48h] [rbp-48h]
  __int64 v41; // [rsp-30h] [rbp-30h]
  __int64 v42; // [rsp-28h] [rbp-28h]
  __int64 v43; // [rsp-8h] [rbp-8h]

  v43 = v3;
  v42 = v4;
  v41 = v2;
  switch ( *(_BYTE *)(a2 + 28) )
  {
    case 0:
      v7 = sub_E5C2C0(a1, a2);
      v8 = *(_QWORD *)(a2 + 8);
      LODWORD(v39[0]) = (((1 << *(_BYTE *)(a2 + 30)) + v7 - 1) & -(1 << *(_BYTE *)(a2 + 30))) - v7;
      if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)v8 + 8LL))(v8) )
        goto LABEL_7;
      if ( (*(_BYTE *)(a2 + 31) & 1) != 0 )
      {
        v22 = (__int64 *)a1[1];
        v11 = *v22;
        v23 = *(__int64 (**)())(*v22 + 80);
        if ( v23 == sub_E5B800 )
        {
          v9 = v39[0];
          v10 = v39[0];
          if ( !LODWORD(v39[0]) )
            return 0;
          goto LABEL_10;
        }
        if ( ((unsigned __int8 (__fastcall *)(__int64 *, __int64, _QWORD *))v23)(v22, a2, v39) )
          return LODWORD(v39[0]);
LABEL_7:
        v9 = v39[0];
        v10 = v39[0];
        if ( !LODWORD(v39[0]) )
          return 0;
        if ( (*(_BYTE *)(a2 + 31) & 1) != 0 )
        {
          v11 = *(_QWORD *)a1[1];
LABEL_10:
          v12 = *(__int64 (**)())(v11 + 176);
          if ( v12 == sub_E5B880 )
          {
            v10 = v9;
          }
          else
          {
            while ( v10 % (unsigned int)v12() )
            {
              v24 = a1[1];
              v10 = LODWORD(v39[0]) + (1LL << *(_BYTE *)(a2 + 30));
              LODWORD(v39[0]) = v10;
              v12 = *(__int64 (**)())(*(_QWORD *)v24 + 176LL);
              if ( v12 == sub_E5B880 )
                goto LABEL_12;
            }
            v10 = v39[0];
          }
        }
      }
      else
      {
        v10 = v39[0];
        if ( !LODWORD(v39[0]) )
          return 0;
      }
LABEL_12:
      result = 0;
      if ( *(_DWORD *)(a2 + 44) < v10 )
        return result;
      return v10;
    case 1:
    case 4:
    case 6:
    case 7:
    case 8:
    case 0xC:
    case 0xD:
      return *(_QWORD *)(a2 + 48);
    case 2:
      v13 = *(_QWORD *)(a2 + 40);
      v37[0] = 0;
      if ( (unsigned __int8)sub_E81940(v13, v37, a1) )
      {
        result = v37[0] * *(unsigned __int8 *)(a2 + 30);
        if ( result >= 0 )
          return result;
        HIBYTE(v40) = 1;
        v14 = *a1;
        v15 = "invalid number of bytes";
      }
      else
      {
        HIBYTE(v40) = 1;
        v14 = *a1;
        v15 = "expected assembly-time absolute expression";
      }
      v16 = *(_QWORD *)(a2 + 48);
      v39[0] = v15;
      LOBYTE(v40) = 3;
      sub_E66880(v14, v16, v39);
      return 0;
    case 3:
      return *(_QWORD *)(a2 + 32);
    case 5:
      v17 = *(_QWORD *)(a2 + 32);
      v30[0] = 0;
      v30[1] = 0;
      v31 = 0;
      v32 = 0;
      if ( !(unsigned __int8)sub_E81960(v17, v30, a1) )
      {
        HIBYTE(v40) = 1;
        v25 = *a1;
        v26 = "expected assembly-time absolute expression";
LABEL_34:
        v27 = *(_QWORD *)(a2 + 40);
        v39[0] = v26;
        LOBYTE(v40) = 3;
        sub_E66880(v25, v27, v39);
        return 0;
      }
      v28 = sub_E5C2C0(a1, a2);
      v18 = v28;
      v19 = v31;
      v29 = v31;
      if ( !v30[0] )
        goto LABEL_23;
      if ( !(unsigned __int8)sub_E5BD10((__int64)a1, *(_QWORD *)(v30[0] + 16LL), (__int64)v37) )
      {
        HIBYTE(v40) = 1;
        v25 = *a1;
        v26 = "expected absolute expression";
        goto LABEL_34;
      }
      v19 = v29 + v37[0];
      v29 += v37[0];
      v18 = v28;
LABEL_23:
      result = v19 - v18;
      if ( (unsigned __int64)result > 0x3FFFFFFF )
      {
        v20 = *a1;
        v33[0] = "invalid .org offset '";
        v33[2] = &v29;
        v34 = 3075;
        v35[0] = v33;
        v35[2] = "' (at offset '";
        v37[0] = v35;
        v37[2] = &v28;
        v40 = 770;
        v21 = *(_QWORD *)(a2 + 40);
        v36 = 770;
        v39[0] = v37;
        v39[2] = "')";
        v38 = 2818;
        sub_E66880(v20, v21, v39);
        return 0;
      }
      return result;
    case 9:
      return *(_QWORD *)(a2 + 40);
    case 0xA:
      return 4;
    case 0xB:
      return *(_QWORD *)(a2 + 72);
    default:
      BUG();
  }
}
