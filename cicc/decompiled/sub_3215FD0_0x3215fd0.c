// Function: sub_3215FD0
// Address: 0x3215fd0
//
__int64 __fastcall sub_3215FD0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rbx
  __int64 *v3; // rdi
  __int64 v4; // rax
  void (*v5)(); // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 *v8; // r8
  __int64 v10; // rsi
  __int16 v11; // dx
  _QWORD *v12; // rdi
  __int64 result; // rax
  __int16 v14; // dx
  __int64 v15; // [rsp-B0h] [rbp-B0h] BYREF
  _QWORD v16[4]; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v17; // [rsp-88h] [rbp-88h]
  _QWORD v18[4]; // [rsp-78h] [rbp-78h] BYREF
  __int16 v19; // [rsp-58h] [rbp-58h]
  _QWORD v20[4]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v21; // [rsp-28h] [rbp-28h]

  switch ( *(_DWORD *)a1 )
  {
    case 0:
      BUG();
    case 1:
      result = (__int64)sub_32152F0((__int64 *)(a1 + 8), (_QWORD **)a2, *(_WORD *)(a1 + 6));
      break;
    case 2:
      result = (__int64)sub_32156E0((__int64 *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 3:
      result = sub_3215490((_QWORD *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 4:
      result = sub_3215550((_QWORD *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 5:
      result = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, __int64))(*(_QWORD *)a2 + 424LL))(
                 a2,
                 *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(**(_QWORD **)(a1 + 8) + 760LL)
                                             + 16LL * *(_QWORD *)(*(_QWORD *)(a1 + 8) + 8LL)
                                             + 8)
                                 + 16LL),
                 0,
                 4);
      break;
    case 6:
      result = sub_3215670(*(_QWORD *)(a1 + 8), a2, *(_WORD *)(a1 + 6));
      break;
    case 7:
      result = sub_3215A30((unsigned __int64 *)(a1 + 8), (_QWORD *)a2, *(_WORD *)(a1 + 6));
      break;
    case 8:
      result = sub_32165A0(*(_QWORD *)(a1 + 8), a2, *(unsigned __int16 *)(a1 + 6));
      break;
    case 9:
      result = sub_3216430(*(_QWORD *)(a1 + 8), a2, *(unsigned __int16 *)(a1 + 6));
      break;
    case 0xA:
      v14 = *(_WORD *)(a1 + 6);
      v8 = (__int64 *)(a1 + 8);
      v10 = *v8;
      if ( v14 == 34 )
        result = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 424LL))(a2, v10, 0, 0);
      else
        result = (__int64)sub_31F0D70(
                            a2,
                            *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 760) + 1288LL) + 32 * v10 + 8),
                            *(_BYTE *)(*(_QWORD *)(a2 + 760) + 3769LL));
      break;
    case 0xB:
      v11 = *(_WORD *)(a1 + 6);
      v12 = *(_QWORD **)(a1 + 8);
      if ( v11 != 8 )
        BUG();
      v2 = v12;
      v3 = *(__int64 **)(a2 + 224);
      if ( *(_BYTE *)(a2 + 488) )
      {
        if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a2 + 200) + 544LL) - 42) <= 1 )
        {
          v4 = *v3;
          v19 = 2818;
          v5 = *(void (**)())(v4 + 120);
          v6 = v2[1];
          v21 = 770;
          v16[1] = v6;
          v16[2] = " [";
          v15 = v6 + 1;
          v18[0] = v16;
          v17 = 773;
          v7 = *v2;
          v18[2] = &v15;
          v20[0] = v18;
          v16[0] = v7;
          v20[2] = " bytes]";
          if ( v5 != nullsub_98 )
          {
            ((void (__fastcall *)(__int64 *, _QWORD *, __int64))v5)(v3, v20, 1);
            v3 = *(__int64 **)(a2 + 224);
          }
        }
      }
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*v3 + 512))(v3, *v2, v2[1]);
      result = sub_31DC9D0(a2, 0);
      break;
    case 0xC:
      result = sub_3215FA0(*(__int64 **)(a1 + 8), (_QWORD **)a2);
      break;
    default:
      return result;
  }
  return result;
}
