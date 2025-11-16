// Function: sub_B9C770
// Address: 0xb9c770
//
__int64 __fastcall sub_B9C770(__int64 *a1, __int64 *a2, __int64 *a3, int a4, char a5)
{
  __int64 v6; // r12
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // rax
  __int64 *v12; // rdx
  unsigned int v13; // esi
  int v14; // eax
  int v15; // eax
  __int64 v16; // [rsp+0h] [rbp-80h]
  int v18; // [rsp+Ch] [rbp-74h]
  __int64 v19; // [rsp+10h] [rbp-70h] BYREF
  __int64 *v20; // [rsp+18h] [rbp-68h] BYREF
  __int64 *v21[4]; // [rsp+20h] [rbp-60h] BYREF
  int v22; // [rsp+40h] [rbp-40h]

  if ( a4 )
  {
    v9 = sub_B97910(16, (unsigned __int64)a3, a4);
    v6 = v9;
    if ( v9 )
    {
      sub_B971C0(v9, (__int64)a1, 5, a4, a2, (__int64)a3, 0, 0);
      *(_DWORD *)(v6 + 4) = 0;
      v19 = v6;
    }
    else
    {
      v19 = 0;
    }
    if ( a4 == 1 )
    {
      sub_B95A20((unsigned __int8 *)v6);
      return v19;
    }
  }
  else
  {
    v21[0] = a2;
    v21[1] = a3;
    v21[2] = 0;
    v21[3] = 0;
    v22 = sub_B75C00(a2, (__int64)a3);
    v6 = sub_B903E0(*a1 + 664, (__int64)v21);
    if ( !v6 && a5 )
    {
      v10 = *a1;
      v18 = v22;
      v11 = sub_B97910(16, (unsigned __int64)a3, 0);
      if ( v11 )
      {
        v16 = v11;
        sub_B971C0(v11, (__int64)a1, 5, 0, a2, (__int64)a3, 0, 0);
        *(_DWORD *)(v16 + 4) = v18;
        v19 = v16;
      }
      else
      {
        v19 = 0;
      }
      if ( !(unsigned __int8)sub_B95B20(v10 + 664, &v19, &v20) )
      {
        v12 = v20;
        v21[0] = v20;
        v13 = *(_DWORD *)(v10 + 688);
        v14 = *(_DWORD *)(v10 + 680);
        ++*(_QWORD *)(v10 + 664);
        v15 = v14 + 1;
        if ( 4 * v15 >= 3 * v13 )
        {
          v13 *= 2;
        }
        else if ( v13 - *(_DWORD *)(v10 + 684) - v15 > v13 >> 3 )
        {
LABEL_15:
          *(_DWORD *)(v10 + 680) = v15;
          if ( *v12 != -4096 )
            --*(_DWORD *)(v10 + 684);
          *v12 = v19;
          return v19;
        }
        sub_B9C570(v10 + 664, v13);
        sub_B95B20(v10 + 664, &v19, v21);
        v12 = v21[0];
        v15 = *(_DWORD *)(v10 + 680) + 1;
        goto LABEL_15;
      }
      return v19;
    }
  }
  return v6;
}
