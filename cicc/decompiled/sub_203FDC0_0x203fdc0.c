// Function: sub_203FDC0
// Address: 0x203fdc0
//
__int64 *__fastcall sub_203FDC0(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  char v7; // al
  __int64 v8; // rdx
  char v9; // di
  __int64 v10; // rdx
  __int64 **v11; // rdi
  __int64 v12; // rsi
  __int64 *v13; // r15
  __int128 v14; // rcx
  __int64 *result; // rax
  __int128 v16; // [rsp+0h] [rbp-160h]
  __int64 *v17; // [rsp+0h] [rbp-160h]
  __int64 *v18; // [rsp+0h] [rbp-160h]
  __int64 v19; // [rsp+10h] [rbp-150h] BYREF
  __int64 v20; // [rsp+18h] [rbp-148h]
  __int64 **v21; // [rsp+20h] [rbp-140h] BYREF
  __int64 v22; // [rsp+28h] [rbp-138h]
  _BYTE v23[304]; // [rsp+30h] [rbp-130h] BYREF

  v7 = *(_BYTE *)(a2 + 88);
  v8 = *(_QWORD *)(a2 + 96);
  LOBYTE(v19) = v7;
  v20 = v8;
  if ( v7 )
  {
    if ( (unsigned __int8)(v7 - 14) > 0x5Fu )
    {
LABEL_3:
      v9 = v19;
      v10 = v20;
      goto LABEL_4;
    }
  }
  else if ( !sub_1F58D20((__int64)&v19) )
  {
    goto LABEL_3;
  }
  v9 = sub_1F7E0F0((__int64)&v19);
LABEL_4:
  LOBYTE(v21) = v9;
  v22 = v10;
  if ( !v9 )
  {
    if ( (sub_1F58D40((__int64)&v21) & 7) == 0 )
      goto LABEL_6;
    return (__int64 *)sub_20B91E0(*a1, a2, a1[1]);
  }
  if ( (sub_2021900(v9) & 7) != 0 )
    return (__int64 *)sub_20B91E0(*a1, a2, a1[1]);
LABEL_6:
  v21 = (__int64 **)v23;
  v22 = 0x1000000000LL;
  if ( (*(_BYTE *)(a2 + 27) & 4) != 0 )
    sub_203F5F0(a1, (__int64)&v21, a2, a3, a4, a5);
  else
    sub_203E950((__int64)a1, (__int64)&v21, a2, a3, a4, a5);
  v11 = v21;
  if ( (unsigned int)v22 == 1 )
  {
    result = *v21;
  }
  else
  {
    v12 = *(_QWORD *)(a2 + 72);
    v13 = (__int64 *)a1[1];
    *(_QWORD *)&v14 = v21;
    *((_QWORD *)&v14 + 1) = (unsigned int)v22;
    v19 = v12;
    if ( v12 )
    {
      *(_QWORD *)&v16 = v21;
      *((_QWORD *)&v16 + 1) = (unsigned int)v22;
      sub_1623A60((__int64)&v19, v12, 2);
      v14 = v16;
    }
    LODWORD(v20) = *(_DWORD *)(a2 + 64);
    result = sub_1D359D0(v13, 2, (__int64)&v19, 1, 0, 0, a3, a4, a5, v14);
    if ( v19 )
    {
      v17 = result;
      sub_161E7C0((__int64)&v19, v19);
      result = v17;
    }
    v11 = v21;
  }
  if ( v11 != (__int64 **)v23 )
  {
    v18 = result;
    _libc_free((unsigned __int64)v11);
    return v18;
  }
  return result;
}
