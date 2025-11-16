// Function: sub_1F13350
// Address: 0x1f13350
//
__int64 __fastcall sub_1F13350(__int64 a1)
{
  __int64 *v2; // r11
  int v3; // ecx
  __int64 v4; // r10
  unsigned int v5; // edi
  unsigned __int64 v6; // r8
  __int64 v7; // rdx
  unsigned int v11; // ecx
  int v12; // eax
  unsigned int v13; // ecx
  unsigned int v14; // r9d
  unsigned int v15; // esi
  int v16; // edx
  __int64 v17; // r10
  unsigned __int64 v18; // r8
  int v19; // ecx
  __int64 v20; // rax
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // r15
  unsigned __int8 v25; // [rsp+1h] [rbp-29h]

  v2 = *(__int64 **)(a1 + 272);
  v3 = *((_DWORD *)v2 + 4);
  if ( v3 )
  {
    v4 = *v2;
    v5 = (unsigned int)(v3 - 1) >> 6;
    v6 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
    v7 = 0;
    while ( 1 )
    {
      _RCX = *(_QWORD *)(v4 + 8 * v7);
      if ( v5 == (_DWORD)v7 )
        _RCX = v6 & *(_QWORD *)(v4 + 8 * v7);
      if ( _RCX )
        break;
      if ( v5 + 1 == ++v7 )
        goto LABEL_7;
    }
    __asm { tzcnt   rcx, rcx }
    v25 = 1;
    if ( (_DWORD)_RCX + ((_DWORD)v7 << 6) != -1 )
    {
      v11 = _RCX + ((_DWORD)v7 << 6);
      do
      {
        if ( *(int *)(*(_QWORD *)(a1 + 264) + 112LL * v11 + 16) <= 0 )
        {
          v25 = 0;
          *(_QWORD *)(**(_QWORD **)(a1 + 272) + 8LL * (v11 >> 6)) &= ~(1LL << v11);
        }
        v12 = *((_DWORD *)v2 + 4);
        v13 = v11 + 1;
        if ( v12 == v13 )
          break;
        v14 = v13 >> 6;
        v15 = (unsigned int)(v12 - 1) >> 6;
        if ( v13 >> 6 > v15 )
          break;
        v16 = 64 - (v13 & 0x3F);
        v17 = *v2;
        v18 = 0xFFFFFFFFFFFFFFFFLL >> v16;
        if ( v16 == 64 )
          v18 = 0;
        v19 = -v12;
        v20 = v14;
        v21 = ~v18;
        v22 = 0xFFFFFFFFFFFFFFFFLL >> v19;
        while ( 1 )
        {
          _RCX = *(_QWORD *)(v17 + 8 * v20);
          if ( v14 == (_DWORD)v20 )
            _RCX = v21 & *(_QWORD *)(v17 + 8 * v20);
          if ( v15 == (_DWORD)v20 )
            _RCX &= v22;
          if ( _RCX )
            break;
          if ( v15 < (unsigned int)++v20 )
            goto LABEL_8;
        }
        __asm { tzcnt   rcx, rcx }
        v11 = ((_DWORD)v20 << 6) + _RCX;
      }
      while ( v11 != -1 );
    }
  }
  else
  {
LABEL_7:
    v25 = 1;
  }
LABEL_8:
  *(_QWORD *)(a1 + 272) = 0;
  return v25;
}
