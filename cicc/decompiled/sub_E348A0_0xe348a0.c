// Function: sub_E348A0
// Address: 0xe348a0
//
__int64 __fastcall sub_E348A0(_BYTE *a1, __int64 a2, _BYTE *a3, __int64 a4)
{
  _BYTE *v4; // r14
  __int64 v6; // rbx
  __int64 result; // rax
  __int64 v8; // r15
  void (__fastcall *v9)(_BYTE *, __int64, __int64); // rax
  void (__fastcall *v10)(_QWORD, _QWORD); // rax
  __int64 v11; // r12
  _BYTE *v12; // rax
  _BYTE v13[16]; // [rsp+0h] [rbp-50h] BYREF
  __int64 (__fastcall *v14)(_BYTE *, _BYTE *, __int64); // [rsp+10h] [rbp-40h]
  void (__fastcall *v15)(_QWORD, _QWORD); // [rsp+18h] [rbp-38h]

  if ( !*((_QWORD *)a1 + 3) )
    goto LABEL_14;
  v4 = a1;
  a1 += 8;
  v6 = (__int64)a3;
  result = (*((__int64 (__fastcall **)(_BYTE *))v4 + 4))(a1);
  if ( *(_QWORD *)v4 )
  {
    v8 = v6 + 32 * a4;
    if ( v6 != v8 )
    {
      while ( 1 )
      {
        v9 = *(void (__fastcall **)(_BYTE *, __int64, __int64))(v6 + 16);
        v14 = 0;
        if ( !v9 )
          break;
        a2 = v6;
        a1 = v13;
        v9(v13, v6, 2);
        v10 = *(void (__fastcall **)(_QWORD, _QWORD))(v6 + 24);
        a3 = *(_BYTE **)(v6 + 16);
        v11 = *(_QWORD *)v4;
        v15 = v10;
        v14 = (__int64 (__fastcall *)(_BYTE *, _BYTE *, __int64))a3;
        if ( !a3 )
          break;
        a2 = v11;
        a1 = v13;
        v10(v13, v11);
        v12 = *(_BYTE **)(v11 + 32);
        if ( (unsigned __int64)v12 < *(_QWORD *)(v11 + 24) )
        {
          a3 = v12 + 1;
          *(_QWORD *)(v11 + 32) = v12 + 1;
          *v12 = 10;
        }
        else
        {
          a2 = 10;
          a1 = (_BYTE *)v11;
          sub_CB5D20(v11, 10);
        }
        result = (__int64)v14;
        if ( v14 )
        {
          a2 = (__int64)v13;
          a1 = v13;
          result = v14(v13, v13, 3);
        }
        v6 += 32;
        if ( v6 == v8 )
          return result;
      }
LABEL_14:
      sub_4263D6(a1, a2, a3);
    }
  }
  return result;
}
