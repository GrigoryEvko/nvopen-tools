// Function: sub_BA25B0
// Address: 0xba25b0
//
__int64 __fastcall sub_BA25B0(__int64 a1, __int64 a2)
{
  unsigned __int8 v3; // al
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r14
  int v7; // r12d
  int v8; // eax
  int v9; // ecx
  unsigned int i; // r13d
  _QWORD **v11; // r15
  _QWORD *v12; // r12
  _BYTE *v13; // rax
  unsigned int v14; // r13d
  unsigned int v16; // esi
  int v17; // eax
  _QWORD *v18; // rdx
  int v19; // eax
  __int64 v20; // rax
  int v21; // [rsp+0h] [rbp-70h]
  int v22; // [rsp+4h] [rbp-6Ch]
  __int64 v23[2]; // [rsp+8h] [rbp-68h] BYREF
  _QWORD *v24; // [rsp+18h] [rbp-58h] BYREF
  _QWORD *v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h] BYREF
  char v27; // [rsp+30h] [rbp-40h]

  v23[0] = a1;
  v3 = *(_BYTE *)(a1 - 16);
  if ( (v3 & 2) != 0 )
    v4 = *(_QWORD *)(a1 - 32);
  else
    v4 = a1 - 16 - 8LL * ((v3 >> 2) & 0xF);
  v25 = *(_QWORD **)(v4 + 8);
  v5 = sub_AF5140(a1, 2u);
  v6 = *(_QWORD *)(a2 + 8);
  v26 = v5;
  v7 = *(_DWORD *)(a2 + 24);
  v27 = *(_BYTE *)(a1 + 1) >> 7;
  if ( v7 )
  {
    v8 = sub_AFB5F0((__int64 *)&v25, &v26);
    v21 = 1;
    v9 = v7 - 1;
    for ( i = (v7 - 1) & v8; ; i = v9 & v14 )
    {
      v11 = (_QWORD **)(v6 + 8LL * i);
      v12 = *v11;
      if ( *v11 == (_QWORD *)-4096LL )
        break;
      if ( v12 != (_QWORD *)-8192LL )
      {
        v22 = v9;
        v13 = sub_A17150((_BYTE *)v12 - 16);
        v9 = v22;
        if ( v25 == *((_QWORD **)v13 + 1) )
        {
          v20 = sub_AF5140((__int64)v12, 2u);
          v9 = v22;
          if ( v26 == v20 && v27 == (unsigned __int8)BYTE1(*v12) >> 7 )
          {
            if ( v11 == (_QWORD **)(*(_QWORD *)(a2 + 8) + 8LL * *(unsigned int *)(a2 + 24)) )
              break;
            return (__int64)v12;
          }
        }
      }
      v14 = v21 + i;
      ++v21;
    }
  }
  if ( !(unsigned __int8)sub_AFE420(a2, v23, &v24) )
  {
    v16 = *(_DWORD *)(a2 + 24);
    v17 = *(_DWORD *)(a2 + 16);
    v18 = v24;
    ++*(_QWORD *)a2;
    v19 = v17 + 1;
    v25 = v18;
    if ( 4 * v19 >= 3 * v16 )
    {
      v16 *= 2;
    }
    else if ( v16 - *(_DWORD *)(a2 + 20) - v19 > v16 >> 3 )
    {
LABEL_14:
      *(_DWORD *)(a2 + 16) = v19;
      if ( *v18 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v18 = v23[0];
      return v23[0];
    }
    sub_B09270(a2, v16);
    sub_AFE420(a2, v23, &v25);
    v18 = v25;
    v19 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_14;
  }
  return v23[0];
}
