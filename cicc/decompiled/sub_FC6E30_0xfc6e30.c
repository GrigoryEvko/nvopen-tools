// Function: sub_FC6E30
// Address: 0xfc6e30
//
_QWORD *__fastcall sub_FC6E30(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rbx
  __int64 v3; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rdi
  unsigned __int64 v8; // rsi
  _QWORD *v9; // rax
  __int64 v10; // r8
  __int64 v11; // rdi
  _QWORD *v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdi
  unsigned int v15; // esi
  _QWORD *v16; // rdx
  _QWORD *v17; // r9
  int v18; // edx
  int v19; // r10d
  int v20; // eax
  int v21; // r10d
  __int64 v22; // [rsp+10h] [rbp-40h]
  _QWORD *v23; // [rsp+10h] [rbp-40h]
  unsigned __int64 v24[2]; // [rsp+20h] [rbp-30h] BYREF
  __int64 v25; // [rsp+30h] [rbp-20h]

  v2 = a2;
  v3 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) + 16LL * *(unsigned int *)(*(_QWORD *)a1 + 16LL));
  if ( *(_BYTE *)(v3 + 64) )
  {
    v13 = *(unsigned int *)(v3 + 56);
    v14 = *(_QWORD *)(v3 + 40);
    if ( (_DWORD)v13 )
    {
      v15 = (v13 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (_QWORD *)(v14 + 16LL * v15);
      v17 = (_QWORD *)*v16;
      if ( v2 == (_QWORD *)*v16 )
      {
LABEL_20:
        if ( v16 != (_QWORD *)(v14 + 16 * v13) )
          return (_QWORD *)v16[1];
      }
      else
      {
        v18 = 1;
        while ( v17 != (_QWORD *)-4096LL )
        {
          v19 = v18 + 1;
          v15 = (v13 - 1) & (v18 + v15);
          v16 = (_QWORD *)(v14 + 16LL * v15);
          v17 = (_QWORD *)*v16;
          if ( v2 == (_QWORD *)*v16 )
            goto LABEL_20;
          v18 = v19;
        }
      }
    }
  }
  if ( !*(_BYTE *)v2 )
    return v2;
  if ( *(_BYTE *)v2 == 1 )
  {
    v5 = *(unsigned int *)(v3 + 24);
    v6 = v2[17];
    if ( (_DWORD)v5 )
    {
      v7 = *(_QWORD *)(v3 + 8);
      v8 = ((_DWORD)v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
      v9 = (_QWORD *)(v7 + (v8 << 6));
      v10 = v9[3];
      if ( v6 == v10 )
      {
LABEL_8:
        if ( v9 != (_QWORD *)(v7 + (v5 << 6)) )
        {
          v11 = v9[7];
          v24[0] = 6;
          v24[1] = 0;
          v25 = v11;
          LOBYTE(v8) = v11 != 0;
          if ( v11 != -4096 && v11 != 0 && v11 != -8192 )
          {
            v8 = v9[5] & 0xFFFFFFFFFFFFFFF8LL;
            sub_BD6050(v24, v8);
            v11 = v25;
            v6 = v2[17];
          }
          if ( v11 == v6 )
            goto LABEL_15;
          if ( v11 )
          {
            v12 = sub_B98A20(v11, v8);
            v6 = v25;
            v2 = v12;
LABEL_15:
            v23 = v2;
            if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
              sub_BD60C0(v24);
            return v23;
          }
          return 0;
        }
      }
      else
      {
        v20 = 1;
        while ( v10 != -4096 )
        {
          v21 = v20 + 1;
          v8 = ((_DWORD)v5 - 1) & (unsigned int)(v20 + v8);
          v9 = (_QWORD *)(v7 + ((unsigned __int64)(unsigned int)v8 << 6));
          v10 = v9[3];
          if ( v6 == v10 )
            goto LABEL_8;
          v20 = v21;
        }
      }
    }
    if ( v6 )
      return 0;
    return v2;
  }
  return (_QWORD *)v22;
}
