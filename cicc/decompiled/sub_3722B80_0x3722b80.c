// Function: sub_3722B80
// Address: 0x3722b80
//
__int64 ***__fastcall sub_3722B80(__int64 *a1)
{
  __int64 v2; // rax
  __int64 ***v3; // rsi
  __int64 ***result; // rax
  __int64 v5; // r12
  __int64 **v6; // r13
  __int64 v7; // rdx
  _QWORD *v8; // rsi
  _QWORD *v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // r10
  __int64 v12; // rdx
  void (*v13)(); // r9
  __int64 v14; // rdi
  __int64 v15; // r8
  __int64 *v16; // rax
  __int64 ***v17; // [rsp+8h] [rbp-108h]
  __int64 v18; // [rsp+18h] [rbp-F8h]
  __int64 *v19; // [rsp+20h] [rbp-F0h]
  __int64 ***v20; // [rsp+28h] [rbp-E8h]
  __int64 **v21; // [rsp+38h] [rbp-D8h]
  __int64 v22; // [rsp+48h] [rbp-C8h] BYREF
  _QWORD v23[4]; // [rsp+50h] [rbp-C0h] BYREF
  __int16 v24; // [rsp+70h] [rbp-A0h]
  _QWORD v25[4]; // [rsp+80h] [rbp-90h] BYREF
  __int16 v26; // [rsp+A0h] [rbp-70h]
  _QWORD v27[4]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v28; // [rsp+D0h] [rbp-40h]

  v2 = a1[1];
  v3 = *(__int64 ****)(v2 + 192);
  result = *(__int64 ****)(v2 + 184);
  v17 = v3;
  v20 = result;
  if ( result != v3 )
  {
    v5 = 0;
    do
    {
      v21 = v20[1];
      if ( v21 != *v20 )
      {
        v6 = *v20;
        do
        {
          v14 = *a1;
          v11 = *(_QWORD *)(*a1 + 224);
          v13 = *(void (**)())(*(_QWORD *)v11 + 120LL);
          v12 = **v6;
          v16 = (__int64 *)(v12 & 0xFFFFFFFFFFFFFFF8LL);
          LODWORD(v12) = (v12 >> 2) & 1;
          v15 = (unsigned int)v12;
          if ( (_DWORD)v12 )
          {
            v8 = (_QWORD *)v16[3];
            v7 = v16[4];
          }
          else
          {
            v7 = *v16;
            v8 = v16 + 4;
          }
          v27[3] = v7;
          v23[0] = "String in Bucket ";
          v24 = 2819;
          v25[2] = ": ";
          v9 = v25;
          v22 = v5;
          v23[2] = &v22;
          v25[0] = v23;
          v26 = 770;
          v27[0] = v25;
          v27[2] = v8;
          v28 = 1282;
          if ( v13 != nullsub_98 )
          {
            v18 = v15;
            v8 = v27;
            v19 = v16;
            ((void (__fastcall *)(__int64, _QWORD *, __int64))v13)(v11, v27, 1);
            v14 = *a1;
            v15 = v18;
            v16 = v19;
          }
          v10 = (__int64)(v16 + 1);
          if ( !v15 )
            ++v16;
          ++v6;
          sub_31F0E70(v14, (__int64)v8, v10, (__int64)v9, v15, (__int64)v13, *v16, v16[1]);
        }
        while ( v21 != v6 );
      }
      v20 += 3;
      ++v5;
      result = v20;
    }
    while ( v17 != v20 );
  }
  return result;
}
