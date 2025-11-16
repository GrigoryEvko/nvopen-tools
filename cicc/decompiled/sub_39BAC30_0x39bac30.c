// Function: sub_39BAC30
// Address: 0x39bac30
//
unsigned __int64 __fastcall sub_39BAC30(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  __int64 *v7; // rcx
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rdi
  __int64 *v12; // r11
  __int64 v13; // rdx
  void (*v14)(); // r10
  unsigned __int64 v15; // [rsp+8h] [rbp-78h]
  __int64 v16; // [rsp+10h] [rbp-70h]
  unsigned __int64 v17; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v18[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 8);
  v17 = 0;
  v2 = *(_QWORD *)(v1 + 176);
  v3 = *(_QWORD *)(v1 + 184);
  result = 0xAAAAAAAAAAAAAAABLL;
  v16 = v2;
  v15 = 0xAAAAAAAAAAAAAAABLL * ((v3 - v2) >> 3);
  if ( v15 )
  {
    v6 = -1;
    for ( result = 0; result < v15; v17 = result )
    {
      v7 = (__int64 *)(v16 + 24 * result);
      v8 = v7[1];
      if ( v8 != *v7 )
      {
        v9 = *v7;
        do
        {
          v10 = *(unsigned int *)(*(_QWORD *)v9 + 8LL);
          if ( !*(_BYTE *)(a1 + 16) || v10 != v6 )
          {
            v11 = *(_QWORD *)a1;
            v12 = *(__int64 **)(*(_QWORD *)a1 + 256LL);
            v13 = *v12;
            v18[0] = "Offset in Bucket ";
            v19 = 2819;
            v14 = *(void (**)())(v13 + 104);
            v18[1] = &v17;
            if ( v14 != nullsub_580 )
            {
              ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v14)(v12, v18, 1);
              v11 = *(_QWORD *)a1;
            }
            sub_396F380(v11);
            v6 = v10;
          }
          v9 += 8;
        }
        while ( v8 != v9 );
      }
      result = v17 + 1;
    }
  }
  return result;
}
