// Function: sub_396D460
// Address: 0x396d460
//
__int64 __fastcall sub_396D460(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // r15
  __int64 v5; // rbx
  unsigned __int64 v6; // r13
  __int64 v7; // rbx
  unsigned int v8; // r13d
  unsigned int v9; // ecx
  unsigned int v10; // edx
  __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // rdx
  unsigned __int64 v18; // r15
  unsigned __int8 *v19; // rax
  int v20; // edx
  unsigned __int8 *v21; // rcx
  unsigned int v22; // eax
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  _QWORD *v25; // rax
  int v26; // eax
  int v27; // eax
  __int64 v28; // rax
  __int64 v29; // [rsp+8h] [rbp-68h]
  __int64 v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 v34; // [rsp+10h] [rbp-60h]
  __int64 v35; // [rsp+18h] [rbp-58h]
  unsigned __int64 v36; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v37; // [rsp+28h] [rbp-48h]
  unsigned int *v38; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v39; // [rsp+38h] [rbp-38h]

  v3 = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)v3 != 13 )
  {
    if ( (_BYTE)v3 == 6 )
    {
      v7 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
      v8 = sub_396D460(v7);
      if ( v8 != -1 )
      {
        v9 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
        if ( v9 == 1 )
          return v8;
        v10 = 1;
        while ( v7 == *(_QWORD *)(a1 + 24 * (v10 - (unsigned __int64)v9)) )
        {
          if ( ++v10 == v9 )
            return v8;
        }
      }
    }
    else if ( (unsigned int)(v3 - 11) <= 1 )
    {
      v19 = (unsigned __int8 *)sub_1595920(a1);
      v8 = *v19;
      v21 = v19;
      if ( v20 == 1 )
        return v8;
      v22 = 1;
      while ( (_BYTE)v8 == v21[v22] )
      {
        if ( v20 == ++v22 )
          return v8;
      }
    }
    return (unsigned int)-1;
  }
  v4 = *(_QWORD *)a1;
  v5 = 1;
  v6 = (unsigned int)sub_15A9FE0(a2, *(_QWORD *)a1);
  while ( 2 )
  {
    switch ( *(_BYTE *)(v4 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v14 = *(_QWORD *)(v4 + 32);
        v4 = *(_QWORD *)(v4 + 24);
        v5 *= v14;
        continue;
      case 1:
        v12 = 16;
        break;
      case 2:
        v12 = 32;
        break;
      case 3:
      case 9:
        v12 = 64;
        break;
      case 4:
        v12 = 80;
        break;
      case 5:
      case 6:
        v12 = 128;
        break;
      case 7:
        v12 = 8 * (unsigned int)sub_15A9520(a2, 0);
        break;
      case 0xB:
        v12 = *(_DWORD *)(v4 + 8) >> 8;
        break;
      case 0xD:
        v12 = 8LL * *(_QWORD *)sub_15A9930(a2, v4);
        break;
      case 0xE:
        v30 = *(_QWORD *)(v4 + 24);
        v35 = *(_QWORD *)(v4 + 32);
        v15 = sub_15A9FE0(a2, v30);
        v16 = v30;
        v17 = 1;
        v18 = v15;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v16 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v28 = *(_QWORD *)(v16 + 32);
              v16 = *(_QWORD *)(v16 + 24);
              v17 *= v28;
              continue;
            case 1:
              v23 = 16;
              break;
            case 2:
              v23 = 32;
              break;
            case 3:
            case 9:
              v23 = 64;
              break;
            case 4:
              v23 = 80;
              break;
            case 5:
            case 6:
              v23 = 128;
              break;
            case 7:
              v33 = v17;
              v26 = sub_15A9520(a2, 0);
              v17 = v33;
              v23 = (unsigned int)(8 * v26);
              break;
            case 0xB:
              v23 = *(_DWORD *)(v16 + 8) >> 8;
              break;
            case 0xD:
              v32 = v17;
              v25 = (_QWORD *)sub_15A9930(a2, v16);
              v17 = v32;
              v23 = 8LL * *v25;
              break;
            case 0xE:
              v29 = v17;
              v31 = *(_QWORD *)(v16 + 32);
              v24 = sub_12BE0A0(a2, *(_QWORD *)(v16 + 24));
              v17 = v29;
              v23 = 8 * v31 * v24;
              break;
            case 0xF:
              v34 = v17;
              v27 = sub_15A9520(a2, *(_DWORD *)(v16 + 8) >> 8);
              v17 = v34;
              v23 = (unsigned int)(8 * v27);
              break;
          }
          break;
        }
        v12 = 8 * v18 * v35 * ((v18 + ((unsigned __int64)(v23 * v17 + 7) >> 3) - 1) / v18);
        break;
      case 0xF:
        v12 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v4 + 8) >> 8);
        break;
    }
    break;
  }
  v13 = v6 * ((v6 + ((unsigned __int64)(v12 * v5 + 7) >> 3) - 1) / v6);
  v8 = -1;
  sub_16A5DD0((__int64)&v36, a1 + 24, 8 * v13);
  if ( sub_16A8E60((__int64)&v36, 8u) )
  {
    sub_16A5D10((__int64)&v38, (__int64)&v36, 8u);
    v8 = (unsigned int)v38;
    if ( v39 > 0x40 )
    {
      v8 = *v38;
      j_j___libc_free_0_0((unsigned __int64)v38);
    }
  }
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  return v8;
}
