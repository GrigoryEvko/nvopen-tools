// Function: sub_311D330
// Address: 0x311d330
//
__int64 __fastcall sub_311D330(__int64 a1, __int64 a2)
{
  int v2; // ebx
  __int64 v3; // rax
  unsigned __int64 v4; // r12
  unsigned __int64 v5; // rbx
  char v6; // al
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r9
  __int64 v10; // r8
  unsigned __int64 v11; // rax
  __int64 v12; // rsi
  __int64 v14; // rax
  __int64 v15; // [rsp+0h] [rbp-70h]
  __int64 v17; // [rsp+18h] [rbp-58h]
  char v18; // [rsp+2Fh] [rbp-41h] BYREF
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v20[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 24LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v2 = *(_DWORD *)(a2 + 8);
  if ( v2 )
  {
    v3 = (unsigned int)(v2 - 1);
    v4 = 0;
    v5 = 1;
    v17 = v3 + 2;
    do
    {
      while ( 1 )
      {
        v6 = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64 *))(*(_QWORD *)a1 + 32LL))(a1, (unsigned int)v4, &v19);
        v10 = (unsigned int)v5;
        if ( v6 )
          break;
        ++v4;
        if ( v17 == ++v5 )
          return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
      }
      v11 = *(unsigned int *)(a2 + 8);
      if ( v11 > v4 || v11 == v5 )
      {
        v12 = *(_QWORD *)a2;
      }
      else if ( v11 > v5 )
      {
        *(_DWORD *)(a2 + 8) = v5;
        v12 = *(_QWORD *)a2;
      }
      else
      {
        if ( *(unsigned int *)(a2 + 12) < v5 )
        {
          sub_C8D5F0(a2, (const void *)(a2 + 16), v5, 0x10u, (unsigned int)v5, v9);
          v10 = (unsigned int)v5;
          v11 = *(unsigned int *)(a2 + 8);
        }
        v8 = a2;
        v12 = *(_QWORD *)a2;
        v14 = *(_QWORD *)a2 + 16 * v11;
        v7 = *(_QWORD *)a2 + 16 * v5;
        if ( v14 != v7 )
        {
          do
          {
            if ( v14 )
            {
              *(_DWORD *)v14 = 0;
              *(_DWORD *)(v14 + 4) = 0;
              *(_QWORD *)(v14 + 8) = 0;
            }
            v14 += 16;
          }
          while ( v7 != v14 );
          v12 = *(_QWORD *)a2;
        }
        *(_DWORD *)(a2 + 8) = v10;
      }
      v15 = v12 + 16 * v4;
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 104LL))(
        a1,
        v12,
        v7,
        v8,
        v10);
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "InstIndex",
             1,
             0,
             &v18,
             v20) )
      {
        sub_311CE30(a1, (unsigned int *)v15);
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v20[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "OpndIndex",
             1,
             0,
             &v18,
             v20) )
      {
        sub_311CE30(a1, (unsigned int *)(v15 + 4));
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v20[0]);
      }
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "OpndHash",
             1,
             0,
             &v18,
             v20) )
      {
        sub_311D010(a1, (_QWORD *)(v15 + 8));
        (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v20[0]);
      }
      ++v4;
      ++v5;
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, v19);
    }
    while ( v17 != v5 );
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 48LL))(a1);
}
