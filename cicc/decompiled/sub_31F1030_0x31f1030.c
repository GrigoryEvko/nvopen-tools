// Function: sub_31F1030
// Address: 0x31f1030
//
__int64 __fastcall sub_31F1030(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 result; // rax
  __int64 *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rax
  void (*v9)(); // rax
  __int64 v10; // [rsp+8h] [rbp-48h]
  _QWORD v11[4]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v12; // [rsp+30h] [rbp-20h]

  v2 = *(_QWORD *)(a2 + 40);
  switch ( *(_BYTE *)(a2 + 32) )
  {
    case 0:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 936LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 v2);
      break;
    case 1:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 920LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 40));
      break;
    case 2:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 928LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 40));
      break;
    case 3:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 896LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 *(_QWORD *)(a2 + 16),
                 v2);
      break;
    case 4:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 888LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 *(_QWORD *)(a2 + 16),
                 *(unsigned int *)(a2 + 24),
                 v2);
      break;
    case 5:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 880LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 v2);
      break;
    case 6:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 872LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 16),
                 v2);
      break;
    case 7:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 864LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 *(_QWORD *)(a2 + 16),
                 v2);
      break;
    case 9:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 960LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 16),
                 v2);
      break;
    case 0xA:
      v5 = *(__int64 **)(a1 + 224);
      v6 = *(_QWORD *)(a2 + 72);
      v7 = *(_QWORD *)(a2 + 80);
      v8 = *v5;
      v12 = 261;
      v11[0] = v6;
      v9 = *(void (**)())(v8 + 120);
      v11[1] = v7;
      if ( v9 != nullsub_98 )
      {
        v10 = v2;
        ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v9)(v5, v11, 1);
        v5 = *(__int64 **)(a1 + 224);
        v2 = v10;
      }
      result = (*(__int64 (__fastcall **)(__int64 *, _QWORD, _QWORD, __int64))(*v5 + 968))(
                 v5,
                 *(_QWORD *)(a2 + 48),
                 *(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 48),
                 v2);
      break;
    case 0xB:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 944LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 v2);
      break;
    case 0xC:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 1000LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 v2);
      break;
    case 0xD:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 1008LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 *(unsigned int *)(a2 + 12),
                 v2);
      break;
    case 0xE:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 1016LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 40));
      break;
    case 0xF:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 1024LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 40));
      break;
    case 0x10:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 224) + 1032LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 40));
      break;
    case 0x11:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 984LL))(
                 *(_QWORD *)(a1 + 224),
                 *(_QWORD *)(a2 + 16),
                 v2);
      break;
    case 0x13:
      result = (*(__int64 (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 224) + 1048LL))(
                 *(_QWORD *)(a1 + 224),
                 *(unsigned int *)(a2 + 8),
                 *(_QWORD *)(a2 + 16),
                 v2);
      break;
    default:
      BUG();
  }
  return result;
}
