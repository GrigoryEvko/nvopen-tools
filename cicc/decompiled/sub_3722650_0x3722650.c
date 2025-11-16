// Function: sub_3722650
// Address: 0x3722650
//
unsigned __int64 __fastcall sub_3722650(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned __int64 result; // rax
  __int64 v6; // r14
  __int64 *v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 *v11; // r11
  __int64 v12; // rax
  void (*v13)(); // rax
  unsigned __int64 v14; // [rsp+0h] [rbp-90h]
  __int64 v15; // [rsp+10h] [rbp-80h]
  unsigned __int64 v16; // [rsp+28h] [rbp-68h] BYREF
  _QWORD v17[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v18; // [rsp+50h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v2 = *(_QWORD *)(v1 + 184);
  v3 = *(_QWORD *)(v1 + 192);
  result = 0xAAAAAAAAAAAAAAABLL;
  v14 = 0xAAAAAAAAAAAAAAABLL * ((v3 - v2) >> 3);
  if ( v14 )
  {
    v6 = -1;
    for ( result = 0; result < v14; v16 = result )
    {
      v7 = (__int64 *)(v2 + 24 * result);
      v15 = v7[1];
      if ( v15 != *v7 )
      {
        v8 = *v7;
        do
        {
          v9 = v6;
          v6 = *(unsigned int *)(*(_QWORD *)v8 + 8LL);
          if ( !*(_BYTE *)(a1 + 16) || v9 != v6 )
          {
            v10 = *(_QWORD *)a1;
            v11 = *(__int64 **)(*(_QWORD *)a1 + 224LL);
            v12 = *v11;
            v17[2] = &v16;
            v17[0] = "Offset in Bucket ";
            v13 = *(void (**)())(v12 + 120);
            v18 = 2819;
            if ( v13 != nullsub_98 )
            {
              ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v13)(v11, v17, 1);
              v10 = *(_QWORD *)a1;
            }
            sub_31DF6B0(v10);
            sub_31DCA50(v10);
          }
          v8 += 8;
        }
        while ( v15 != v8 );
      }
      result = v16 + 1;
    }
  }
  return result;
}
