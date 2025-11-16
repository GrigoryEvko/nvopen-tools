// Function: sub_39A7AD0
// Address: 0x39a7ad0
//
__int64 __fastcall sub_39A7AD0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdx
  __int64 v5; // r13
  bool v6; // r14
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // r14
  void *v12; // rcx
  size_t v13; // rdx
  size_t v14; // r8
  char v15; // [rsp+Ch] [rbp-44h]
  __int64 v16; // [rsp+18h] [rbp-38h]

  v4 = *(unsigned int *)(a3 + 8);
  v5 = *(_QWORD *)(a3 + 8 * (3 - v4));
  if ( v5 )
  {
    v6 = sub_39A1AB0(*(_QWORD *)(a3 + 8 * (3 - v4)));
    if ( (unsigned __int16)sub_398C0A0(a1[25]) > 2u )
      sub_39A6760(a1, a2, v5, 73);
    if ( (unsigned __int16)sub_398C0A0(a1[25]) > 3u && (*(_BYTE *)(a3 + 31) & 1) != 0 )
      sub_39A34D0((__int64)a1, a2, 109);
    v4 = *(unsigned int *)(a3 + 8);
  }
  else
  {
    v6 = 0;
  }
  result = 4 - v4;
  v8 = *(_QWORD *)(a3 + 8 * (4 - v4));
  if ( v8 )
  {
    result = *(unsigned int *)(v8 + 8);
    if ( (_DWORD)result )
    {
      v16 = *(unsigned int *)(v8 + 8);
      v9 = 0;
      v15 = v6;
      while ( 1 )
      {
        v10 = *(_QWORD *)(v8 + 8 * (v9 - result));
        if ( v10 && *(_BYTE *)v10 == 10 )
        {
          v11 = sub_39A5A90((__int64)a1, 40, a2, 0);
          v12 = *(void **)(v10 - 8LL * *(unsigned int *)(v10 + 8));
          if ( v12 )
          {
            v12 = (void *)sub_161E970(*(_QWORD *)(v10 - 8LL * *(unsigned int *)(v10 + 8)));
            v14 = v13;
          }
          else
          {
            v14 = 0;
          }
          sub_39A3F30(a1, v11, 3, v12, v14);
          result = sub_39A37F0((__int64)a1, v11, v15, *(_QWORD *)(v10 + 24));
        }
        if ( v16 == ++v9 )
          break;
        result = *(unsigned int *)(v8 + 8);
      }
    }
  }
  return result;
}
