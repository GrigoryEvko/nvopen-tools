// Function: sub_39A8670
// Address: 0x39a8670
//
unsigned __int8 *__fastcall sub_39A8670(__int64 *a1, __int64 a2)
{
  unsigned __int8 *v2; // r14
  unsigned __int8 *v3; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r14
  void *v8; // rcx
  size_t v9; // rdx
  size_t v10; // r8
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  int v15; // r8d
  _DWORD v16[13]; // [rsp+Ch] [rbp-34h] BYREF

  if ( !a2 )
    return 0;
  v2 = sub_39A81B0((__int64)a1, *(unsigned __int8 **)(a2 + 8 * (1LL - *(unsigned int *)(a2 + 8))));
  v3 = sub_39A23D0((__int64)a1, (unsigned __int8 *)a2);
  if ( !v3 )
  {
    v5 = sub_39A5A90((__int64)a1, *(_WORD *)(a2 + 2), (__int64)v2, (unsigned __int8 *)a2);
    v6 = *(unsigned int *)(a2 + 8);
    v3 = (unsigned __int8 *)v5;
    v7 = *(_QWORD *)(a2 + 8 * (3 - v6));
    v8 = *(void **)(a2 + 8 * (2 - v6));
    if ( v8 )
    {
      v8 = (void *)sub_161E970(*(_QWORD *)(a2 + 8 * (2 - v6)));
      v10 = v9;
    }
    else
    {
      v10 = 0;
    }
    sub_39A3F30(a1, (__int64)v3, 3, v8, v10);
    sub_39A6760(a1, (__int64)v3, v7, 73);
    sub_39A3790((__int64)a1, (__int64)v3, a2);
    sub_39A34D0((__int64)a1, (__int64)v3, 63);
    sub_39A34D0((__int64)a1, (__int64)v3, 60);
    v11 = *(_DWORD *)(a2 + 28) & 3;
    switch ( v11 )
    {
      case 2:
        v16[0] = 65547;
        sub_39A3560((__int64)a1, (__int64 *)v3 + 1, 50, (__int64)v16, 2);
        break;
      case 1:
        v16[0] = 65547;
        sub_39A3560((__int64)a1, (__int64 *)v3 + 1, 50, (__int64)v16, 3);
        break;
      case 3:
        v16[0] = 65547;
        sub_39A3560((__int64)a1, (__int64 *)v3 + 1, 50, (__int64)v16, 1);
        break;
    }
    v12 = *(_QWORD *)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8)));
    if ( v12 )
    {
      v13 = *(_QWORD *)(v12 + 136);
      if ( v13 )
      {
        if ( *(_BYTE *)(v13 + 16) != 13
          || (sub_39A5150((__int64)a1, (__int64)v3, v13, v7),
              (v14 = *(_QWORD *)(a2 + 8 * (4LL - *(unsigned int *)(a2 + 8)))) != 0)
          && (v13 = *(_QWORD *)(v14 + 136)) != 0 )
        {
          if ( *(_BYTE *)(v13 + 16) == 14 )
            sub_39A50A0(a1, (__int64)v3, v13);
        }
      }
    }
    if ( (unsigned __int16)sub_398C0A0(a1[25]) > 4u )
    {
      v15 = *(_DWORD *)(a2 + 48) >> 3;
      if ( v15 )
      {
        v16[0] = 65551;
        sub_39A3560((__int64)a1, (__int64 *)v3 + 1, 136, (__int64)v16, v15 & 0x1FFFFFFF);
      }
    }
  }
  return v3;
}
