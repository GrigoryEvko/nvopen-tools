// Function: sub_2AA91C0
// Address: 0x2aa91c0
//
unsigned __int64 __fastcall sub_2AA91C0(int *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  unsigned int v9; // eax
  unsigned int *v10; // r11
  unsigned int v11; // eax
  unsigned int v12; // r10d
  __int64 v13; // r8
  unsigned __int64 result; // rax
  __int64 v15; // [rsp+0h] [rbp-40h] BYREF
  int v16; // [rsp+8h] [rbp-38h]
  __int64 v17; // [rsp+10h] [rbp-30h] BYREF
  int v18; // [rsp+18h] [rbp-28h]
  __int64 v19; // [rsp+20h] [rbp-20h] BYREF
  __int64 v20; // [rsp+28h] [rbp-18h]

  v8 = *(_QWORD *)(*((_QWORD *)a1 + 1) + 48LL);
  v9 = *a1;
  if ( *(_BYTE *)(v8 + 108) && *(_DWORD *)(v8 + 100) )
  {
    v19 = a3;
    v20 = a4;
    v18 = 0;
    v17 = (v9 != 0) + (v9 - (v9 != 0)) / a2;
    sub_2AA9150((__int64)&v19, (__int64)&v17);
    return v19;
  }
  else
  {
    v19 = a5;
    v20 = a6;
    v18 = 0;
    v17 = v9 % a2;
    sub_2AA9150((__int64)&v19, (__int64)&v17);
    v11 = *v10;
    v16 = 0;
    v19 = a3;
    v20 = a4;
    v15 = v11 / v12;
    sub_2AA9150((__int64)&v19, (__int64)&v15);
    result = v19 + v13;
    if ( __OFADD__(v19, v13) )
    {
      result = 0x7FFFFFFFFFFFFFFFLL;
      if ( v13 <= 0 )
        return 0x8000000000000000LL;
    }
  }
  return result;
}
