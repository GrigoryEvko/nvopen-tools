// Function: sub_1468AA0
// Address: 0x1468aa0
//
__int64 __fastcall sub_1468AA0(_QWORD *a1, __int64 a2)
{
  __int16 v2; // ax
  __int64 v3; // rdi
  unsigned int v4; // eax
  unsigned int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // r13d
  unsigned int v13; // r15d
  unsigned int v14; // eax
  __int64 v15; // rax
  unsigned int v16; // eax
  __int64 v17; // rcx
  unsigned int v18; // r13d
  unsigned int v19; // r15d
  unsigned int v20; // eax
  __int64 v21; // rax
  int v22; // r13d
  unsigned int v23; // r15d
  unsigned int v24; // eax
  __int64 v25; // rax
  int v26; // r13d
  unsigned int v27; // r15d
  unsigned int v28; // eax
  __int64 v29; // rax
  int v30; // r13d
  unsigned int v31; // r15d
  unsigned int v32; // eax
  __int64 v33; // r15
  __int64 v34; // r12
  int v35; // eax
  int v37; // [rsp+Ch] [rbp-54h]
  __int64 v38; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v39; // [rsp+18h] [rbp-48h]
  __int64 v40[8]; // [rsp+20h] [rbp-40h] BYREF

  v2 = *(_WORD *)(a2 + 24);
  if ( v2 )
  {
    switch ( v2 )
    {
      case 1:
        LODWORD(_R12) = sub_1456C90((__int64)a1, *(_QWORD *)(a2 + 40));
        v9 = sub_14687F0((__int64)a1, *(_QWORD *)(a2 + 32));
        if ( v9 < (unsigned int)_R12 )
          LODWORD(_R12) = v9;
        return (unsigned int)_R12;
      case 2:
      case 3:
        LODWORD(_R12) = sub_14687F0((__int64)a1, *(_QWORD *)(a2 + 32));
        v10 = sub_1456040(*(_QWORD *)(a2 + 32));
        if ( (unsigned int)_R12 == sub_1456C90((__int64)a1, v10) )
          LODWORD(_R12) = sub_1456C90((__int64)a1, *(_QWORD *)(a2 + 40));
        return (unsigned int)_R12;
      case 4:
        LODWORD(_R12) = sub_14687F0((__int64)a1, **(_QWORD **)(a2 + 32));
        v11 = *(_QWORD *)(a2 + 40);
        v12 = v11;
        if ( (_DWORD)_R12 )
        {
          if ( (_DWORD)v11 == 1 )
            return (unsigned int)_R12;
          v13 = 1;
          while ( 1 )
          {
            v14 = sub_14687F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * v13));
            if ( (unsigned int)_R12 > v14 )
              LODWORD(_R12) = v14;
            ++v13;
            if ( !(_DWORD)_R12 )
              break;
            if ( v12 == v13 )
              return (unsigned int)_R12;
          }
        }
        break;
      case 5:
        LODWORD(_R12) = sub_14687F0((__int64)a1, **(_QWORD **)(a2 + 32));
        v15 = sub_1456040(**(_QWORD **)(a2 + 32));
        v16 = sub_1456C90((__int64)a1, v15);
        v17 = *(_QWORD *)(a2 + 40);
        v18 = v16;
        v37 = v17;
        if ( (_DWORD)_R12 != v16 && (_DWORD)v17 != 1 )
        {
          v19 = 1;
          do
          {
            v20 = _R12 + sub_14687F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * v19));
            LODWORD(_R12) = v20;
            if ( v18 <= v20 )
              LODWORD(_R12) = v18;
            ++v19;
          }
          while ( v18 > v20 && v37 != v19 );
        }
        return (unsigned int)_R12;
      case 7:
        LODWORD(_R12) = sub_14687F0((__int64)a1, **(_QWORD **)(a2 + 32));
        v21 = *(_QWORD *)(a2 + 40);
        v22 = v21;
        if ( (_DWORD)_R12 )
        {
          if ( (_DWORD)v21 == 1 )
            return (unsigned int)_R12;
          v23 = 1;
          while ( 1 )
          {
            v24 = sub_14687F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * v23));
            if ( (unsigned int)_R12 > v24 )
              LODWORD(_R12) = v24;
            ++v23;
            if ( !(_DWORD)_R12 )
              break;
            if ( v22 == v23 )
              return (unsigned int)_R12;
          }
        }
        break;
      case 9:
        LODWORD(_R12) = sub_14687F0((__int64)a1, **(_QWORD **)(a2 + 32));
        v25 = *(_QWORD *)(a2 + 40);
        v26 = v25;
        if ( (_DWORD)_R12 )
        {
          if ( (_DWORD)v25 == 1 )
            return (unsigned int)_R12;
          v27 = 1;
          while ( 1 )
          {
            v28 = sub_14687F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * v27));
            if ( (unsigned int)_R12 > v28 )
              LODWORD(_R12) = v28;
            ++v27;
            if ( !(_DWORD)_R12 )
              break;
            if ( v26 == v27 )
              return (unsigned int)_R12;
          }
        }
        break;
      case 8:
        LODWORD(_R12) = sub_14687F0((__int64)a1, **(_QWORD **)(a2 + 32));
        v29 = *(_QWORD *)(a2 + 40);
        v30 = v29;
        if ( (_DWORD)_R12 )
        {
          if ( (_DWORD)v29 == 1 )
            return (unsigned int)_R12;
          v31 = 1;
          while ( 1 )
          {
            v32 = sub_14687F0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * v31));
            if ( (unsigned int)_R12 > v32 )
              LODWORD(_R12) = v32;
            ++v31;
            if ( !(_DWORD)_R12 )
              break;
            if ( v30 == v31 )
              return (unsigned int)_R12;
          }
        }
        break;
      default:
        LODWORD(_R12) = 0;
        if ( v2 == 10 )
        {
          v33 = a1[7];
          v34 = a1[6];
          v35 = sub_1632FA0(*(_QWORD *)(a1[3] + 40LL));
          sub_14C2530((unsigned int)&v38, *(_QWORD *)(a2 - 8), v35, 0, v34, 0, v33, 0);
          if ( v39 > 0x40 )
          {
            LODWORD(_R12) = sub_16A58F0(&v38);
          }
          else
          {
            _R12 = ~v38;
            if ( v38 == -1 )
              LODWORD(_R12) = 64;
            else
              __asm { tzcnt   r12, r12 }
          }
          sub_135E100(v40);
          sub_135E100(&v38);
        }
        return (unsigned int)_R12;
    }
    LODWORD(_R12) = 0;
    return (unsigned int)_R12;
  }
  v3 = *(_QWORD *)(a2 + 32);
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
  {
    _RDX = *(_QWORD *)(v3 + 24);
    LODWORD(_R12) = 64;
    __asm { tzcnt   rcx, rdx }
    if ( _RDX )
      LODWORD(_R12) = _RCX;
    if ( (unsigned int)_R12 > v4 )
      LODWORD(_R12) = *(_DWORD *)(v3 + 32);
    return (unsigned int)_R12;
  }
  return sub_16A58A0(v3 + 24);
}
