// Function: sub_24F2810
// Address: 0x24f2810
//
__int64 *__fastcall sub_24F2810(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 *v7; // rdx
  __int64 *result; // rax
  unsigned __int8 *v9; // r14
  __int64 v10; // rax
  int v11; // edx
  unsigned __int8 *v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r15
  int v17; // r15d
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // r15
  __int64 v21; // r13
  __int64 v22; // rdi
  __int64 v23; // r12
  _QWORD *v24; // rax
  _QWORD *v25; // r14
  __int64 v26; // rdx
  __int64 v27; // rdx
  __int64 v28; // rdx
  char v29; // al
  __int64 *v30; // rcx
  __int64 i; // rax
  __int64 v32; // rax
  __int64 *v33; // [rsp+10h] [rbp-A0h]
  __int64 v34; // [rsp+18h] [rbp-98h]
  unsigned __int8 *v35; // [rsp+20h] [rbp-90h]
  __int64 v36; // [rsp+28h] [rbp-88h]
  __int64 v37; // [rsp+30h] [rbp-80h]
  __int64 *v38; // [rsp+38h] [rbp-78h]
  __int64 *v39; // [rsp+40h] [rbp-70h]
  unsigned __int8 *v40; // [rsp+48h] [rbp-68h]
  _QWORD v41[4]; // [rsp+50h] [rbp-60h] BYREF
  __int16 v42; // [rsp+70h] [rbp-40h]

  v1 = **(_QWORD **)(*(_QWORD *)(sub_B43CB0(**(_QWORD **)(a1 + 16)) + 24) + 16LL);
  if ( *(_BYTE *)(v1 + 8) == 15 )
  {
    v2 = *(unsigned int *)(v1 + 12) - 1LL;
    v36 = *(_QWORD *)(v1 + 16) + 8LL;
  }
  else
  {
    v36 = 0;
    v2 = 0;
  }
  v3 = *(_QWORD *)(a1 + 16);
  v4 = *(_QWORD *)(*(_QWORD *)(v3 + 328) + 24LL);
  v5 = *(_QWORD *)(v4 + 16);
  v6 = ((8LL * *(unsigned int *)(v4 + 12) - 8) >> 3) - 1;
  v7 = *(__int64 **)(v3 + 120);
  v37 = v6;
  result = &v7[*(unsigned int *)(v3 + 128)];
  v33 = result;
  if ( v7 != result )
  {
    v38 = v7;
    v39 = (__int64 *)(v36 + 8 * v2);
    do
    {
      v9 = (unsigned __int8 *)*v38;
      v10 = *(_QWORD *)(*v38 - 32);
      if ( !v10 || *(_BYTE *)v10 || *(_QWORD *)(v10 + 24) != *((_QWORD *)v9 + 10) )
        BUG();
      if ( *(_DWORD *)(v10 + 36) != 62 )
        sub_C64ED0("coro.id.retcon.* must be paired with coro.suspend.retcon", 1u);
      v11 = *v9;
      v12 = &v9[-32 * (*((_DWORD *)v9 + 1) & 0x7FFFFFF)];
      if ( v11 == 40 )
      {
        v13 = -32 - 32LL * (unsigned int)sub_B491D0(*v38);
      }
      else
      {
        v13 = -32;
        if ( v11 != 85 )
        {
          if ( v11 != 34 )
            goto LABEL_65;
          v13 = -96;
        }
      }
      if ( (v9[7] & 0x80u) != 0 )
      {
        v14 = sub_BD2BC0((__int64)v9);
        v16 = v14 + v15;
        if ( (v9[7] & 0x80u) == 0 )
        {
          if ( (unsigned int)(v16 >> 4) )
LABEL_65:
            BUG();
        }
        else if ( (unsigned int)((v16 - sub_BD2BC0((__int64)v9)) >> 4) )
        {
          if ( (v9[7] & 0x80u) == 0 )
            goto LABEL_65;
          v17 = *(_DWORD *)(sub_BD2BC0((__int64)v9) + 8);
          if ( (v9[7] & 0x80u) == 0 )
            BUG();
          v18 = sub_BD2BC0((__int64)v9);
          v13 -= 32LL * (unsigned int)(*(_DWORD *)(v18 + v19 - 4) - v17);
        }
      }
      v20 = (__int64 *)v36;
      v40 = &v9[v13];
      if ( v12 != &v9[v13] && (__int64 *)v36 != v39 )
      {
        v34 = v5;
        v35 = v9;
        v21 = (__int64)(v9 + 24);
        do
        {
          v22 = *(_QWORD *)(*(_QWORD *)v12 + 8LL);
          if ( *v20 != v22 )
          {
            if ( !(unsigned __int8)sub_B50B40(v22, *v20) )
              sub_C64ED0("argument to coro.suspend.retcon does not match corresponding prototype function result", 1u);
            v23 = *(_QWORD *)v12;
            v42 = 257;
            v24 = sub_BD2C40(72, unk_3F10A14);
            v25 = v24;
            if ( v24 )
            {
              sub_B51BF0((__int64)v24, v23, *v20, (__int64)v41, v21, 0);
              if ( *(_QWORD *)v12 )
              {
                v26 = *((_QWORD *)v12 + 1);
                **((_QWORD **)v12 + 2) = v26;
                if ( v26 )
                  *(_QWORD *)(v26 + 16) = *((_QWORD *)v12 + 2);
              }
              *(_QWORD *)v12 = v25;
              v27 = v25[2];
              *((_QWORD *)v12 + 1) = v27;
              if ( v27 )
                *(_QWORD *)(v27 + 16) = v12 + 8;
              *((_QWORD *)v12 + 2) = v25 + 2;
              v25[2] = v12;
            }
            else if ( *(_QWORD *)v12 )
            {
              v32 = *((_QWORD *)v12 + 1);
              **((_QWORD **)v12 + 2) = v32;
              if ( v32 )
                *(_QWORD *)(v32 + 16) = *((_QWORD *)v12 + 2);
              *(_QWORD *)v12 = 0;
            }
          }
          v12 += 32;
          ++v20;
        }
        while ( v12 != v40 && v39 != v20 );
        v9 = v35;
        v5 = v34;
      }
      if ( v40 != v12 || v39 != v20 )
        sub_C64ED0("wrong number of arguments to coro.suspend.retcon", 1u);
      v28 = *((_QWORD *)v9 + 1);
      v41[0] = v28;
      v29 = *(_BYTE *)(v28 + 8);
      if ( v29 == 7 )
      {
        if ( v37 )
LABEL_55:
          sub_C64ED0("wrong number of results from coro.suspend.retcon", 1u);
      }
      else
      {
        if ( v29 == 15 )
        {
          v30 = *(__int64 **)(v28 + 16);
          if ( *(_DWORD *)(v28 + 12) != v37 )
            goto LABEL_55;
          if ( !v37 )
            goto LABEL_47;
          v28 = *v30;
        }
        else
        {
          if ( v37 != 1 )
            goto LABEL_55;
          v30 = v41;
        }
        for ( i = 0; ; v28 = v30[i] )
        {
          if ( *(_QWORD *)(v5 + 8 * i + 16) != v28 )
            sub_C64ED0("result from coro.suspend.retcon does not match corresponding prototype function param", 1u);
          if ( ++i == v37 )
            break;
        }
      }
LABEL_47:
      result = ++v38;
    }
    while ( v33 != v38 );
  }
  return result;
}
