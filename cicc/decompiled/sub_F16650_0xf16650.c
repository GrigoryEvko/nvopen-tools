// Function: sub_F16650
// Address: 0xf16650
//
void __fastcall sub_F16650(__int64 a1, __int64 a2, unsigned __int8 *a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 *v7; // rbx
  int v8; // edx
  __int64 v9; // rdi

  v5 = *(_QWORD *)(a2 + 16);
  if ( v5 )
  {
    while ( 1 )
    {
      v6 = v5;
      v5 = *(_QWORD *)(v5 + 8);
      v7 = *(__int64 **)(v6 + 24);
      if ( a3 == (unsigned __int8 *)v7 )
        goto LABEL_4;
      v8 = *(unsigned __int8 *)v7;
      switch ( v8 )
      {
        case 59:
          sub_F162A0(a1, (__int64)v7, a2);
          sub_F15FC0(*(_QWORD *)(a1 + 40), (__int64)v7);
          goto LABEL_4;
        case 86:
          sub_BD28A0(v7 - 8, v7 - 4);
          sub_B47280((__int64)v7);
LABEL_4:
          if ( !v5 )
            return;
          break;
        case 31:
          sub_B4CC70((__int64)v7);
          v9 = *(_QWORD *)(a1 + 184);
          if ( !v9 )
            goto LABEL_4;
          sub_FF0720(v9, v7[5]);
          if ( !v5 )
            return;
          break;
        default:
          BUG();
      }
    }
  }
}
