// Function: sub_AE1890
// Address: 0xae1890
//
_QWORD *__fastcall sub_AE1890(_QWORD *a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5, __int64 a6, unsigned int a7)
{
  unsigned int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned int v12; // ebx
  const char *v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rdx
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rax
  _QWORD v20[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v21; // [rsp+20h] [rbp-70h] BYREF
  __int64 v22; // [rsp+30h] [rbp-60h] BYREF
  __int64 v23; // [rsp+38h] [rbp-58h]
  const char *v24; // [rsp+40h] [rbp-50h]
  __int16 v25; // [rsp+50h] [rbp-40h]

  if ( a3 )
  {
    if ( (unsigned __int8)sub_C93C90(a2, a3, 10, &v22) || v22 != (unsigned int)v22 || (v22 & 0xFFFF0000) != 0 )
    {
      v22 = a5;
      v15 = " alignment must be a 16-bit integer";
      v25 = 773;
      v23 = a6;
    }
    else if ( (_WORD)v22 )
    {
      if ( (v22 & 7) == 0 && (unsigned __int16)v22 > 7u )
      {
        v18 = (unsigned __int16)v22 >> 3;
        if ( ((unsigned int)v18 & ((_DWORD)v18 - 1)) == 0 )
        {
          _BitScanReverse64(&v19, v18);
          *a4 = 63 - (v19 ^ 0x3F);
          *a1 = 1;
          return a1;
        }
      }
      v22 = a5;
      v25 = 773;
      v15 = " alignment must be a power of two times the byte width";
      v23 = a6;
    }
    else
    {
      if ( (_BYTE)a7 )
      {
        *a4 = 0;
        *a1 = 1;
        return a1;
      }
      v22 = a5;
      v15 = " alignment must be non-zero";
      v25 = 773;
      v23 = a6;
    }
    v24 = v15;
    v16 = ((__int64 (*)(void))sub_C63BB0)();
    v11 = v17;
    v12 = v16;
  }
  else
  {
    v22 = a5;
    v25 = 773;
    v23 = a6;
    v24 = " alignment component cannot be empty";
    v9 = sub_C63BB0(a1, 773, 0, a4, a7);
    v11 = v10;
    v12 = v9;
  }
  sub_CA0F50(v20, &v22);
  sub_C63F00(a1, v20, v12, v11);
  if ( (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0], v21 + 1);
  return a1;
}
