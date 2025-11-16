// Function: sub_828890
// Address: 0x828890
//
__int64 __fastcall sub_828890(__int64 *a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  const __m128i *v6; // rax
  __int64 v7; // rsi
  __m128i *v8; // rdi
  __m128i *v9; // rax
  __int64 v10; // rt0
  int v11; // [rsp+4h] [rbp-2Ch] BYREF
  _BYTE v12[40]; // [rsp+8h] [rbp-28h] BYREF

  result = sub_8D2E30(a2);
  if ( (_DWORD)result )
  {
    result = sub_8D2E30(*a1);
    if ( (_DWORD)result )
    {
      result = sub_8DB6D0(*a1, a2);
      if ( result )
      {
LABEL_5:
        *a1 = result;
        return result;
      }
      v4 = sub_8D46C0(*a1);
      v5 = sub_8D46C0(a2);
      if ( (unsigned int)sub_8D2600(v4) || (unsigned int)sub_8D2600(v5) )
      {
        v6 = (const __m128i *)sub_72CBE0();
        v7 = v5;
        v8 = sub_73CA70(v6, v4);
LABEL_9:
        v9 = sub_73CA70(v8, v7);
        result = sub_72D2E0(v9);
        goto LABEL_5;
      }
      result = sub_8D5EF0(*a1, a2, &v11, v12);
      if ( (_DWORD)result )
      {
        if ( v11 )
        {
          v10 = v5;
          v5 = v4;
          v4 = v10;
        }
        v7 = v5;
        v8 = (__m128i *)v4;
        goto LABEL_9;
      }
    }
  }
  return result;
}
