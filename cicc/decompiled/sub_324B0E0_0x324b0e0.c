// Function: sub_324B0E0
// Address: 0x324b0e0
//
__int64 __fastcall sub_324B0E0(__int64 *a1, __int64 a2, __int64 a3)
{
  unsigned __int8 v5; // al
  __int64 v6; // rdx
  __int64 v7; // rdi
  const void *v8; // rax
  size_t v9; // rdx
  __int64 result; // rax
  unsigned __int64 **v11; // r12
  __int64 v12; // r8
  __int64 v13; // r8
  int v14; // [rsp+Ch] [rbp-24h]

  v5 = *(_BYTE *)(a3 - 16);
  if ( (v5 & 2) != 0 )
    v6 = *(_QWORD *)(a3 - 32);
  else
    v6 = a3 - 16 - 8LL * ((v5 >> 2) & 0xF);
  v7 = *(_QWORD *)(v6 + 16);
  if ( v7 )
  {
    v8 = (const void *)sub_B91420(v7);
    if ( v9 )
      sub_324AD70(a1, a2, 3, v8, v9);
  }
  result = sub_AF18C0(a3);
  if ( (_WORD)result != 59 )
  {
    v11 = (unsigned __int64 **)(a2 + 8);
    if ( (unsigned __int16)sub_AF18C0(a3) != 18 )
    {
      v14 = 65547;
      sub_3249A20(a1, v11, 62, 65547, *(unsigned int *)(a3 + 44));
    }
    BYTE2(v14) = 0;
    sub_3249A20(a1, v11, 11, v14, *(_QWORD *)(a3 + 24) >> 3);
    result = *(unsigned int *)(a3 + 20);
    if ( (result & 0x8000000) != 0 )
    {
      BYTE2(v14) = 0;
      v13 = 1;
    }
    else
    {
      if ( (result & 0x10000000) == 0 )
        goto LABEL_11;
      BYTE2(v14) = 0;
      v13 = 2;
    }
    result = sub_3249A20(a1, v11, 101, v14, v13);
LABEL_11:
    v12 = *(unsigned int *)(a3 + 40);
    if ( (_DWORD)v12 )
    {
      BYTE2(v14) = 0;
      return sub_3249A20(a1, v11, 15883, v14, v12);
    }
  }
  return result;
}
