// Function: sub_39AD590
// Address: 0x39ad590
//
void __fastcall sub_39AD590(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  int v7; // r14d
  const char *v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  unsigned int *v14; // rax
  _QWORD v15[2]; // [rsp-58h] [rbp-58h] BYREF
  _QWORD v16[2]; // [rsp-48h] [rbp-48h] BYREF
  __int16 v17; // [rsp-38h] [rbp-38h]

  if ( *(_QWORD *)(a1 + 32) )
  {
    if ( *(_BYTE *)(a1 + 26) || *(_BYTE *)(a1 + 24) )
    {
      v2 = *(_QWORD *)(a1 + 8);
      v3 = *(_QWORD *)(v2 + 264);
      v4 = *(_QWORD *)v3;
      if ( (*(_BYTE *)(*(_QWORD *)v3 + 18LL) & 8) != 0 )
      {
        v5 = sub_15E38F0(*(_QWORD *)v3);
        v6 = sub_1649C60(v5);
        v7 = sub_14DD7D0(v6);
        (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 968LL))(
          *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
          0);
        if ( v7 == 9 )
        {
          if ( *(_BYTE *)(a1 + 24) && !*(_BYTE *)(*(_QWORD *)(a1 + 32) + 184LL) )
          {
            v8 = sub_1649960(v4);
            if ( v9 )
            {
              if ( *v8 == 1 )
              {
                --v9;
                ++v8;
              }
            }
            v15[0] = v8;
            v10 = *(_QWORD *)(a1 + 8);
            v15[1] = v9;
            v11 = *(_QWORD *)(v10 + 248);
            v17 = 1283;
            v16[0] = "$cppxdata$";
            v16[1] = v15;
            v12 = sub_38BF510(v11, (__int64)v16);
            v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL);
            v14 = (unsigned int *)sub_39ACBF0(a1, v12);
            sub_38DDD30(v13, v14);
          }
        }
        else if ( v7 == 8 && *(_BYTE *)(v3 + 523) && !*(_BYTE *)(*(_QWORD *)(a1 + 32) + 183LL) )
        {
          sub_39AD190(a1, (__int64 *)v3);
        }
      }
      else
      {
        (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(v2 + 256) + 968LL))(*(_QWORD *)(v2 + 256), 0);
      }
      (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 160LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
        *(_QWORD *)(a1 + 40),
        0);
      (*(void (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 880LL))(
        *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
        0);
    }
    *(_QWORD *)(a1 + 32) = 0;
  }
}
