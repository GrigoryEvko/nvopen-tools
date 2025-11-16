// Function: sub_39BAB10
// Address: 0x39bab10
//
_QWORD *__fastcall sub_39BAB10(__int64 a1)
{
  __int64 v1; // rax
  _QWORD *v2; // rcx
  _QWORD *result; // rax
  int v5; // r12d
  __int64 v6; // rsi
  __int64 v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r13
  int v10; // r8d
  __int64 v11; // rdi
  __int64 *v12; // r11
  __int64 v13; // rdx
  void (*v14)(); // r10
  _QWORD *v15; // [rsp+8h] [rbp-88h]
  int v16; // [rsp+14h] [rbp-7Ch]
  _QWORD *v17; // [rsp+18h] [rbp-78h]
  __int64 v18; // [rsp+20h] [rbp-70h]
  _QWORD v19[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v20; // [rsp+50h] [rbp-40h]

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_QWORD **)(v1 + 184);
  result = *(_QWORD **)(v1 + 176);
  v15 = v2;
  v17 = result;
  if ( result != v2 )
  {
    v5 = 0;
    v6 = -1;
    do
    {
      v7 = v17[1];
      if ( v7 != *v17 )
      {
        v8 = *v17;
        do
        {
          v9 = *(unsigned int *)(*(_QWORD *)v8 + 8LL);
          v10 = *(_DWORD *)(*(_QWORD *)v8 + 8LL);
          if ( !*(_BYTE *)(a1 + 16) || v9 != v6 )
          {
            v11 = *(_QWORD *)a1;
            LODWORD(v18) = v5;
            v12 = *(__int64 **)(*(_QWORD *)a1 + 256LL);
            v13 = *v12;
            v19[0] = "Hash in Bucket ";
            v20 = 2307;
            v14 = *(void (**)())(v13 + 104);
            v19[1] = v18;
            if ( v14 != nullsub_580 )
            {
              v16 = v10;
              ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v14)(v12, v19, 1);
              v11 = *(_QWORD *)a1;
              v10 = v16;
            }
            sub_396F340(v11, v10);
            v6 = v9;
          }
          v8 += 8;
        }
        while ( v7 != v8 );
      }
      v17 += 3;
      ++v5;
      result = v17;
    }
    while ( v15 != v17 );
  }
  return result;
}
