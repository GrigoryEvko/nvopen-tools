// Function: sub_29067B0
// Address: 0x29067b0
//
__int64 __fastcall sub_29067B0(__int64 *a1, __int64 *a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdi
  int v13; // eax
  int v15; // edx
  int v16; // r10d
  _QWORD v17[3]; // [rsp+0h] [rbp-60h] BYREF
  int v18; // [rsp+18h] [rbp-48h]
  __int64 v19; // [rsp+20h] [rbp-40h] BYREF
  __int64 v20; // [rsp+28h] [rbp-38h]
  __int64 v21; // [rsp+30h] [rbp-30h]

  v2 = *a1;
  v3 = sub_2906530(*a2, *(_QWORD *)v2, *(_QWORD *)(v2 + 8));
  v4 = **(_QWORD **)(v2 + 16);
  v5 = *(unsigned int *)(v4 + 24);
  v6 = *(_QWORD *)(v4 + 8);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( v3 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v6 + 16 * v5) )
      {
        v10 = *(_QWORD *)(v4 + 32);
        v11 = v10 + ((unsigned __int64)*((unsigned int *)v8 + 2) << 6);
        if ( v11 != ((unsigned __int64)*(unsigned int *)(v4 + 40) << 6) + v10 )
        {
          sub_28FF950((__int64)v17, v11 + 8);
          goto LABEL_6;
        }
      }
    }
    else
    {
      v15 = 1;
      while ( v9 != -4096 )
      {
        v16 = v15 + 1;
        v7 = (v5 - 1) & (v15 + v7);
        v8 = (__int64 *)(v6 + 16LL * v7);
        v9 = *v8;
        if ( v3 == *v8 )
          goto LABEL_3;
        v15 = v16;
      }
    }
  }
  v17[0] = 0;
  v17[1] = 0;
  v17[2] = v3;
  if ( v3 )
  {
    if ( v3 == -4096 || v3 == -8192 )
    {
      v18 = 1;
      v19 = 0;
      v20 = 0;
      v21 = v3;
    }
    else
    {
      sub_BD73F0((__int64)v17);
      v18 = 1;
      v19 = 0;
      v20 = 0;
      v21 = v3;
      sub_BD73F0((__int64)&v19);
    }
  }
  else
  {
    v18 = 1;
    v19 = 0;
    v20 = 0;
    v21 = 0;
  }
LABEL_6:
  v12 = *(_QWORD *)(v2 + 24);
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 != 2 )
  {
    if ( v13 )
    {
      if ( v18 && (v18 == 2 || *(_QWORD *)(v12 + 48) != v21) )
      {
        *(_DWORD *)(v12 + 24) = 2;
        sub_FC7530((_QWORD *)(v12 + 32), 0);
      }
    }
    else
    {
      *(_DWORD *)(v12 + 24) = v18;
      sub_FC7530((_QWORD *)(v12 + 32), v21);
    }
  }
  sub_D68D70(&v19);
  return sub_D68D70(v17);
}
