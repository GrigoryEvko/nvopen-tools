// Function: sub_1113590
// Address: 0x1113590
//
__int64 __fastcall sub_1113590(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v6; // r13
  __int64 *v7; // r14
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 *v12; // rbx
  __int64 v13; // r13
  __int64 v14; // r12
  __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rdx
  unsigned int v18; // esi
  _BYTE v19[32]; // [rsp+0h] [rbp-90h] BYREF
  __int16 v20; // [rsp+20h] [rbp-70h]
  _BYTE v21[32]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v22; // [rsp+50h] [rbp-40h]

  v2 = a2;
  if ( !sub_B4DE30(**(_QWORD **)a1) )
  {
    if ( !**(_DWORD **)(a1 + 8) )
      goto LABEL_5;
    __asm { tzcnt   eax, eax }
    if ( _EAX )
    {
LABEL_5:
      v6 = sub_AD62B0(*(_QWORD *)(a2 + 8));
      v7 = *(__int64 **)(*(_QWORD *)(a1 + 16) + 32LL);
      _RAX = *(_DWORD **)(a1 + 8);
      v22 = 257;
      LODWORD(_RAX) = *_RAX;
      __asm { tzcnt   esi, eax }
      _RSI = (int)_RSI;
      if ( !(_DWORD)_RAX )
        _RSI = 32;
      v10 = sub_AD64C0(*(_QWORD *)(v6 + 8), _RSI, 0);
      v11 = sub_F94560(v7, v6, v10, (__int64)v21, 0);
      v12 = *(__int64 **)(*(_QWORD *)(a1 + 16) + 32LL);
      v20 = 257;
      v13 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v12[10] + 16LL))(
              v12[10],
              28,
              v2,
              v11);
      if ( !v13 )
      {
        v22 = 257;
        v13 = sub_B504D0(28, v2, v11, (__int64)v21, 0, 0);
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v12[11] + 16LL))(
          v12[11],
          v13,
          v19,
          v12[7],
          v12[8]);
        v14 = 16LL * *((unsigned int *)v12 + 2);
        v15 = *v12;
        v16 = v15 + v14;
        while ( v16 != v15 )
        {
          v17 = *(_QWORD *)(v15 + 8);
          v18 = *(_DWORD *)v15;
          v15 += 16;
          sub_B99FD0(v13, v18, v17);
        }
      }
      return v13;
    }
  }
  return v2;
}
