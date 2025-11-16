// Function: sub_1403F30
// Address: 0x1403f30
//
__int64 __fastcall sub_1403F30(_QWORD *a1, _QWORD *a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 (__fastcall *v9)(__int64, _QWORD *, _QWORD *, __int64, size_t); // r15
  _QWORD *v10; // rdx
  __int64 v11; // rcx
  _BYTE *v12; // r10
  size_t v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rax
  _QWORD *v18; // rdi
  size_t n; // [rsp+8h] [rbp-78h]
  _BYTE *src; // [rsp+10h] [rbp-70h]
  void (__fastcall *v21)(__int64, __int64, char *, __int64); // [rsp+18h] [rbp-68h]
  _QWORD *v22; // [rsp+28h] [rbp-58h] BYREF
  _QWORD *v23; // [rsp+30h] [rbp-50h] BYREF
  _QWORD *v24; // [rsp+38h] [rbp-48h]
  _QWORD v25[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = *a2;
  *a1 = 0;
  result = (*(__int64 (__fastcall **)(_QWORD *))(v5 + 120))(a2);
  if ( result )
  {
    result = (*(__int64 (__fastcall **)(_QWORD *))(*a2 + 120LL))(a2);
    *(_QWORD *)(result + 8) = a3;
    return result;
  }
  if ( a3 )
  {
    v7 = sub_163A1D0(a2, a2);
    v8 = sub_163A340(v7, a2[2]);
    v9 = *(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD *, __int64, size_t))(*(_QWORD *)a3 + 32LL);
    v21 = *(void (__fastcall **)(__int64, __int64, char *, __int64))(*(_QWORD *)a3 + 40LL);
    v12 = (_BYTE *)(*(__int64 (__fastcall **)(_QWORD *))(*a2 + 16LL))(a2);
    v13 = (size_t)v10;
    if ( !v12 )
    {
      v24 = 0;
      v23 = v25;
      v15 = v25;
      LOBYTE(v25[0]) = 0;
      goto LABEL_11;
    }
    v22 = v10;
    v14 = v10;
    v23 = v25;
    if ( (unsigned __int64)v10 > 0xF )
    {
      n = (size_t)v10;
      src = v12;
      v17 = sub_22409D0(&v23, &v22, 0);
      v12 = src;
      v13 = n;
      v23 = (_QWORD *)v17;
      v18 = (_QWORD *)v17;
      v25[0] = v22;
    }
    else
    {
      if ( v10 == (_QWORD *)1 )
      {
        LOBYTE(v25[0]) = *v12;
        v10 = v25;
LABEL_9:
        v24 = v14;
        *((_BYTE *)v14 + (_QWORD)v10) = 0;
        v15 = v23;
LABEL_11:
        v16 = v9(a3, v15, v10, v11, v13);
        v21(a3, v16, "NVVM", 1);
        if ( v23 != v25 )
          j_j___libc_free_0(v23, v25[0] + 1LL);
        result = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a3 + 56LL))(a3, 0);
        if ( v8 )
        {
          if ( *(_BYTE *)(v8 + 41) )
            result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a3 + 56LL))(a3, 1);
        }
        *a1 = a3;
        return result;
      }
      if ( !v10 )
      {
        v10 = v25;
        goto LABEL_9;
      }
      v18 = v25;
    }
    memcpy(v18, v12, v13);
    v14 = v22;
    v10 = v23;
    goto LABEL_9;
  }
  return result;
}
