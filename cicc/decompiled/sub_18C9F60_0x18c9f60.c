// Function: sub_18C9F60
// Address: 0x18c9f60
//
__int64 __fastcall sub_18C9F60(__int64 a1, __int64 a2)
{
  bool v2; // al
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rax
  _BYTE *v7; // rax
  const char *v8; // [rsp+0h] [rbp-30h] BYREF
  char v9; // [rsp+10h] [rbp-20h]
  char v10; // [rsp+11h] [rbp-1Fh]

  v2 = sub_18C8CE0(a2);
  *(_BYTE *)(a1 + 328) = v2;
  if ( v2 )
  {
    *(_QWORD *)(a1 + 248) = a2;
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 264) = 0;
    *(_QWORD *)(a1 + 272) = 0;
    *(_QWORD *)(a1 + 280) = 0;
    *(_QWORD *)(a1 + 288) = 0;
    *(_QWORD *)(a1 + 296) = 0;
    *(_QWORD *)(a1 + 304) = 0;
    *(_QWORD *)(a1 + 312) = 0;
    *(_QWORD *)(a1 + 320) = 0;
    *(_QWORD *)(a1 + 336) = 0;
    v10 = 1;
    v8 = "clang.arc.retainAutoreleasedReturnValueMarker";
    v9 = 3;
    v4 = sub_1632310(a2, (__int64)&v8);
    v5 = v4;
    if ( v4 )
    {
      if ( (unsigned int)sub_161F520(v4) == 1 )
      {
        v6 = sub_161F530(v5, 0);
        if ( *(_DWORD *)(v6 + 8) == 1 )
        {
          v7 = *(_BYTE **)(v6 - 8);
          if ( !*v7 )
            *(_QWORD *)(a1 + 336) = v7;
        }
      }
    }
  }
  return 0;
}
