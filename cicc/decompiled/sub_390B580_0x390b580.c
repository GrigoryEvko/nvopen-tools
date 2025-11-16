// Function: sub_390B580
// Address: 0x390b580
//
__int64 __fastcall sub_390B580(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v6; // rcx
  unsigned int v7; // r13d
  unsigned int (*i)(void); // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  const char *v11; // rax
  unsigned __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  __int64 v18; // rdi
  const char *v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v22; // [rsp+8h] [rbp-C8h] BYREF
  _QWORD v23[2]; // [rsp+10h] [rbp-C0h] BYREF
  __int16 v24; // [rsp+20h] [rbp-B0h]
  _QWORD v25[2]; // [rsp+30h] [rbp-A0h] BYREF
  __int16 v26; // [rsp+40h] [rbp-90h]
  _QWORD v27[2]; // [rsp+50h] [rbp-80h] BYREF
  __int16 v28; // [rsp+60h] [rbp-70h]
  _QWORD v29[2]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v30; // [rsp+80h] [rbp-50h]
  _QWORD v31[2]; // [rsp+90h] [rbp-40h] BYREF
  __int64 v32; // [rsp+A0h] [rbp-30h]
  int v33; // [rsp+A8h] [rbp-28h]

  switch ( *(_BYTE *)(a3 + 16) )
  {
    case 0:
      v6 = (unsigned int)sub_38D01B0((__int64)a2, a3);
      v7 = *(_DWORD *)(a3 + 48) * (((unsigned __int64)*(unsigned int *)(a3 + 48) + v6 - 1) / *(unsigned int *)(a3 + 48))
         - v6;
      result = 0;
      if ( v7 )
      {
        if ( (*(_BYTE *)(a3 + 52) & 1) != 0 )
        {
          for ( i = *(unsigned int (**)(void))(**(_QWORD **)(a1 + 8) + 112LL);
                (char *)i != (char *)sub_390A070;
                i = *(unsigned int (**)(void))(**(_QWORD **)(a1 + 8) + 112LL) )
          {
            if ( !(v7 % i()) )
              break;
            v7 += *(_DWORD *)(a3 + 48);
          }
        }
        result = 0;
        if ( v7 <= *(_DWORD *)(a3 + 68) )
          return v7;
      }
      return result;
    case 1:
    case 2:
    case 4:
    case 6:
    case 8:
    case 0xC:
      return *(unsigned int *)(a3 + 72);
    case 3:
      v9 = *(_QWORD *)(a3 + 64);
      v29[0] = 0;
      if ( !sub_38CF2A0(v9, v29, a2) )
      {
        BYTE1(v32) = 1;
        v10 = *(_QWORD *)a1;
        v11 = "expected assembly-time absolute expression";
        goto LABEL_12;
      }
      result = v29[0] * *(unsigned __int8 *)(a3 + 56);
      if ( result < 0 )
      {
        BYTE1(v32) = 1;
        v10 = *(_QWORD *)a1;
        v11 = "invalid number of bytes";
LABEL_12:
        v12 = *(_QWORD *)(a3 + 72);
        v31[0] = v11;
        LOBYTE(v32) = 3;
        sub_38BE3D0(v10, v12, (__int64)v31);
        return 0;
      }
      return result;
    case 5:
      v13 = *(_QWORD *)(a3 + 48);
      v31[0] = 0;
      v31[1] = 0;
      v32 = 0;
      v33 = 0;
      if ( !(unsigned __int8)sub_38CF2F0(v13, (__int64)v31, a2) )
      {
        HIBYTE(v30) = 1;
        v18 = *(_QWORD *)a1;
        v19 = "expected assembly-time absolute expression";
LABEL_27:
        v20 = *(_QWORD *)(a3 + 64);
        v29[0] = v19;
        LOBYTE(v30) = 3;
        sub_38BE3D0(v18, v20, (__int64)v29);
        return 0;
      }
      v21 = sub_38D01B0((__int64)a2, a3);
      v14 = v21;
      v15 = v32;
      v22 = v32;
      if ( !v31[0] )
        goto LABEL_17;
      if ( !(unsigned __int8)sub_38D0480(a2, *(_QWORD *)(v31[0] + 24LL), v27) )
      {
        HIBYTE(v30) = 1;
        v18 = *(_QWORD *)a1;
        v19 = "expected absolute expression";
        goto LABEL_27;
      }
      v15 = v22 + v27[0];
      v22 += v27[0];
      v14 = v21;
LABEL_17:
      result = v15 - v14;
      if ( (unsigned __int64)result > 0x3FFFFFFF )
      {
        v16 = *(_QWORD *)a1;
        v23[0] = "invalid .org offset '";
        v23[1] = &v22;
        v24 = 3075;
        v25[0] = v23;
        v25[1] = "' (at offset '";
        v27[0] = v25;
        v27[1] = &v21;
        v30 = 770;
        v17 = *(_QWORD *)(a3 + 64);
        v26 = 770;
        v29[0] = v27;
        v29[1] = "')";
        v28 = 2818;
        sub_38BE3D0(v16, v17, (__int64)v29);
        return 0;
      }
      return result;
    case 7:
      return *(unsigned int *)(a3 + 64);
    case 9:
      return *(_QWORD *)(a3 + 64);
    case 0xA:
      return 4;
    case 0xB:
      return *(unsigned int *)(a3 + 88);
  }
}
