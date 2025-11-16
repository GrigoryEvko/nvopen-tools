// Function: sub_A70200
// Address: 0xa70200
//
_BOOL8 __fastcall sub_A70200(__int64 a1, __int64 a2)
{
  switch ( a2 )
  {
    case 10LL:
      return *(_QWORD *)a1 == 0x696C61636F6C6C61LL && *(_WORD *)(a1 + 8) == 28263
          || *(_QWORD *)a1 == 0x65677265766E6F63LL && *(_WORD *)(a1 + 8) == 29806
          || *(_QWORD *)a1 == 0x6968656E696C6E69LL && *(_WORD *)(a1 + 8) == 29806
          || *(_QWORD *)a1 == 0x61626C6C61636F6ELL && *(_WORD *)(a1 + 8) == 27491
          || *(_QWORD *)a1 == 0x6568635F66636F6ELL && *(_WORD *)(a1 + 8) == 27491
          || *(_QWORD *)a1 == 0x7973617466697773LL && *(_WORD *)(a1 + 8) == 25454
          || *(_QWORD *)a1 == 0x7272657466697773LL && *(_WORD *)(a1 + 8) == 29295
          || *(_QWORD *)a1 == 0x757465726C6C6977LL && *(_WORD *)(a1 + 8) == 28274
          || *(_QWORD *)a1 == 0x6174736E67696C61LL && *(_WORD *)(a1 + 8) == 27491;
    case 8LL:
      if ( *(_QWORD *)a1 == 0x727470636F6C6C61LL
        || *(_QWORD *)a1 == 0x656E696C6E696F6ELL
        || *(_QWORD *)a1 == 0x6E72757465726F6ELL
        || *(_QWORD *)a1 == 0x646E69776E756F6ELL
        || *(_QWORD *)a1 == 0x677562656474706FLL
        || *(_QWORD *)a1 == 0x656E6F6E64616572LL
        || *(_QWORD *)a1 == 0x796C6E6F64616572LL
        || *(_QWORD *)a1 == 0x64656E7275746572LL
        || *(_QWORD *)a1 == 0x7066746369727473LL
        || *(_QWORD *)a1 == 0x656C626174697277LL
        || *(_QWORD *)a1 == 0x61636F6C6C616E69LL
        || *(_QWORD *)a1 == 0x7365727574706163LL )
      {
        return 1;
      }
      break;
    case 12LL:
      if ( *(_QWORD *)a1 == 0x6E69737961776C61LL && *(_DWORD *)(a1 + 8) == 1701734764
        || *(_QWORD *)a1 == 0x676F72707473756DLL && *(_DWORD *)(a1 + 8) == 1936942450
        || *(_QWORD *)a1 == 0x74616C7563657073LL && *(_DWORD *)(a1 + 8) == 1701601889
        || *(_QWORD *)a1 == 0x636F6C6C61657270LL && *(_DWORD *)(a1 + 8) == 1684370529
        || *(_QWORD *)a1 == 0x725F656C61637376LL && *(_DWORD *)(a1 + 8) == 1701277281 )
      {
        return 1;
      }
      break;
    case 7LL:
      if ( *(_DWORD *)a1 == 1818850658 && *(_WORD *)(a1 + 4) == 26996 && *(_BYTE *)(a1 + 6) == 110
        || *(_DWORD *)a1 == 1936615789 && *(_WORD *)(a1 + 4) == 31337 && *(_BYTE *)(a1 + 6) == 101
        || *(_DWORD *)a1 == 1818324846 && *(_WORD *)(a1 + 4) == 24937 && *(_BYTE *)(a1 + 6) == 115
        || *(_DWORD *)a1 == 1701670766 && *(_WORD *)(a1 + 4) == 26482 && *(_BYTE *)(a1 + 6) == 101
        || *(_DWORD *)a1 == 1853189998 && *(_WORD *)(a1 + 4) == 25956 && *(_BYTE *)(a1 + 6) == 102
        || *(_DWORD *)a1 == 1852731246 && *(_WORD *)(a1 + 4) == 27765 && *(_BYTE *)(a1 + 6) == 108
        || *(_DWORD *)a1 == 1937010799 && *(_WORD *)(a1 + 4) == 31337 && *(_BYTE *)(a1 + 6) == 101
        || *(_DWORD *)a1 == 1853124719 && *(_WORD *)(a1 + 4) == 28271 && *(_BYTE *)(a1 + 6) == 101
        || *(_DWORD *)a1 == 1852270963 && *(_WORD *)(a1 + 4) == 30821 && *(_BYTE *)(a1 + 6) == 116
        || *(_DWORD *)a1 == 1869768058 && *(_WORD *)(a1 + 4) == 30821 && *(_BYTE *)(a1 + 6) == 116
        || *(_DWORD *)a1 == 1635022709 && *(_WORD *)(a1 + 4) == 27746 && *(_BYTE *)(a1 + 6) == 101 )
      {
        return 1;
      }
      break;
    case 4LL:
      if ( *(_DWORD *)a1 == 1684828003 || *(_DWORD *)a1 == 1953719662 || *(_DWORD *)a1 == 1952805491 )
        return 1;
      break;
    case 31LL:
      return memcmp((const void *)a1, "coro_only_destroy_when_complete", 0x1Fu) == 0;
    case 15LL:
      if ( *(_QWORD *)a1 == 0x696C655F6F726F63LL
        && *(_DWORD *)(a1 + 8) == 1935631716
        && *(_WORD *)(a1 + 12) == 26209
        && *(_BYTE *)(a1 + 14) == 101
        || *(_QWORD *)a1 == 0x63696C706D696F6ELL
        && *(_DWORD *)(a1 + 8) == 1818653801
        && *(_WORD *)(a1 + 12) == 24943
        && *(_BYTE *)(a1 + 14) == 116
        || *(_QWORD *)a1 == 0x657A6974696E6173LL
        && *(_DWORD *)(a1 + 8) == 1835363679
        && *(_WORD *)(a1 + 12) == 24948
        && *(_BYTE *)(a1 + 14) == 103
        || *(_QWORD *)a1 == 0x657A6974696E6173LL
        && *(_DWORD *)(a1 + 8) == 1835363679
        && *(_WORD *)(a1 + 12) == 29295
        && *(_BYTE *)(a1 + 14) == 121
        || *(_QWORD *)a1 == 0x657A6974696E6173LL
        && *(_DWORD *)(a1 + 8) == 1919448159
        && *(_WORD *)(a1 + 12) == 24933
        && *(_BYTE *)(a1 + 14) == 100
        || *(_QWORD *)a1 == 0x6163776F64616873LL
        && *(_DWORD *)(a1 + 8) == 1953721452
        && *(_WORD *)(a1 + 12) == 25441
        && *(_BYTE *)(a1 + 14) == 107
        || *(_QWORD *)a1 == 0x6572656665726564LL
        && *(_DWORD *)(a1 + 8) == 1634034542
        && *(_WORD *)(a1 + 12) == 27746
        && *(_BYTE *)(a1 + 14) == 101
        || *(_QWORD *)a1 == 0x2D73666E692D6F6ELL
        && *(_DWORD *)(a1 + 8) == 1831694438
        && *(_WORD *)(a1 + 12) == 29793
        && *(_BYTE *)(a1 + 14) == 104
        || *(_QWORD *)a1 == 0x2D736E616E2D6F6ELL
        && *(_DWORD *)(a1 + 8) == 1831694438
        && *(_WORD *)(a1 + 12) == 29793
        && *(_BYTE *)(a1 + 14) == 104 )
      {
        return 1;
      }
      break;
    case 14LL:
      if ( *(_QWORD *)a1 == 0x5F6E6F5F64616564LL && *(_DWORD *)(a1 + 8) == 1769434741 && *(_WORD *)(a1 + 12) == 25710
        || *(_QWORD *)a1 == 0x2D706D756A2D6F6ELL && *(_DWORD *)(a1 + 8) == 1818386804 && *(_WORD *)(a1 + 12) == 29541
        || *(_QWORD *)a1 == 0x662D656661736E75LL && *(_DWORD *)(a1 + 8) == 1634545008 && *(_WORD *)(a1 + 12) == 26740 )
      {
        return 1;
      }
      break;
    case 33LL:
      if ( !(*(_QWORD *)a1 ^ 0x5F656C6261736964LL | *(_QWORD *)(a1 + 8) ^ 0x657A6974696E6173LL)
        && !(*(_QWORD *)(a1 + 16) ^ 0x757274736E695F72LL | *(_QWORD *)(a1 + 24) ^ 0x6F697461746E656DLL)
        && *(_BYTE *)(a1 + 32) == 110 )
      {
        return 1;
      }
      break;
    case 19LL:
      if ( !(*(_QWORD *)a1 ^ 0x745F7465725F6E66LL | *(_QWORD *)(a1 + 8) ^ 0x7478655F6B6E7568LL)
        && *(_WORD *)(a1 + 16) == 29285
        && *(_BYTE *)(a1 + 18) == 110
        || !(*(_QWORD *)a1 ^ 0x6974696E61736F6ELL | *(_QWORD *)(a1 + 8) ^ 0x7265766F635F657ALL)
        && *(_WORD *)(a1 + 16) == 26465
        && *(_BYTE *)(a1 + 18) == 101
        || !(*(_QWORD *)a1 ^ 0x662D786F72707061LL | *(_QWORD *)(a1 + 8) ^ 0x6D2D70662D636E75LL)
        && *(_WORD *)(a1 + 16) == 29793
        && *(_BYTE *)(a1 + 18) == 104 )
      {
        return 1;
      }
      break;
    case 3LL:
      if ( *(_WORD *)a1 == 28520 && *(_BYTE *)(a1 + 2) == 116 || *(_WORD *)a1 == 29555 && *(_BYTE *)(a1 + 2) == 112 )
        return 1;
      break;
    case 16LL:
      if ( !(*(_QWORD *)a1 ^ 0x705F646972627968LL | *(_QWORD *)(a1 + 8) ^ 0x656C626168637461LL)
        || !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x737365726464615FLL)
        || !(*(_QWORD *)a1 ^ 0x6C616D726F6E6564LL | *(_QWORD *)(a1 + 8) ^ 0x6874616D2D70662DLL) )
      {
        return 1;
      }
      break;
    case 6LL:
      if ( *(_DWORD *)a1 == 1634561385 && *(_WORD *)(a1 + 4) == 26482
        || *(_DWORD *)a1 == 1919315822 && *(_WORD *)(a1 + 4) == 25957
        || *(_DWORD *)a1 == 2037608302 && *(_WORD *)(a1 + 4) == 25454
        || *(_DWORD *)a1 == 1919972211 && *(_WORD *)(a1 + 4) == 29029
        || *(_DWORD *)a1 == 1869440365 && *(_WORD *)(a1 + 4) == 31090 )
      {
        return 1;
      }
      break;
    case 5LL:
      if ( *(_DWORD *)a1 == 1701998185 && *(_BYTE *)(a1 + 4) == 103
        || *(_DWORD *)a1 == 1701536110 && *(_BYTE *)(a1 + 4) == 100
        || *(_DWORD *)a1 == 2019913582 && *(_BYTE *)(a1 + 4) == 116
        || *(_DWORD *)a1 == 1702000994 && *(_BYTE *)(a1 + 4) == 102
        || *(_DWORD *)a1 == 1635154274 && *(_BYTE *)(a1 + 4) == 108
        || *(_DWORD *)a1 == 1734962273 && *(_BYTE *)(a1 + 4) == 110
        || *(_DWORD *)a1 == 1735287154 && *(_BYTE *)(a1 + 4) == 101 )
      {
        return 1;
      }
      break;
    case 9LL:
      if ( *(_QWORD *)a1 == 0x6C626174706D756ALL && *(_BYTE *)(a1 + 8) == 101
        || *(_QWORD *)a1 == 0x69746C6975626F6ELL && *(_BYTE *)(a1 + 8) == 110
        || *(_QWORD *)a1 == 0x6C69666F72706F6ELL && *(_BYTE *)(a1 + 8) == 101
        || *(_QWORD *)a1 == 0x7372756365726F6ELL && *(_BYTE *)(a1 + 8) == 101
        || *(_QWORD *)a1 == 0x6E6F7A6465726F6ELL && *(_BYTE *)(a1 + 8) == 101
        || *(_QWORD *)a1 == 0x6361747365666173LL && *(_BYTE *)(a1 + 8) == 107
        || *(_QWORD *)a1 == 0x6E6F727473707373LL && *(_BYTE *)(a1 + 8) == 103
        || *(_QWORD *)a1 == 0x6C65737466697773LL && *(_BYTE *)(a1 + 8) == 102
        || *(_QWORD *)a1 == 0x6C6E6F6574697277LL && *(_BYTE *)(a1 + 8) == 121
        || *(_QWORD *)a1 == 0x6E696B636F6C6C61LL && *(_BYTE *)(a1 + 8) == 100
        || *(_QWORD *)a1 == 0x7A6973636F6C6C61LL && *(_BYTE *)(a1 + 8) == 101
        || *(_QWORD *)a1 == 0x73616C6370666F6ELL && *(_BYTE *)(a1 + 8) == 115 )
      {
        return 1;
      }
      break;
    case 18LL:
      if ( !(*(_QWORD *)a1 ^ 0x6772657669646F6ELL | *(_QWORD *)(a1 + 8) ^ 0x72756F7365636E65LL)
        && *(_WORD *)(a1 + 16) == 25955
        || !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x657264646177685FLL)
        && *(_WORD *)(a1 + 16) == 29555
        || !(*(_QWORD *)a1 ^ 0x6572702D7373656CLL | *(_QWORD *)(a1 + 8) ^ 0x6D70662D65736963LL)
        && *(_WORD *)(a1 + 16) == 25697
        || !(*(_QWORD *)a1 ^ 0x706D61732D657375LL | *(_QWORD *)(a1 + 8) ^ 0x69666F72702D656CLL)
        && *(_WORD *)(a1 + 16) == 25964 )
      {
        return 1;
      }
      break;
    case 11LL:
      if ( *(_QWORD *)a1 == 0x63696C7075646F6ELL && *(_WORD *)(a1 + 8) == 29793 && *(_BYTE *)(a1 + 10) == 101
        || *(_QWORD *)a1 == 0x62797A616C6E6F6ELL && *(_WORD *)(a1 + 8) == 28265 && *(_BYTE *)(a1 + 10) == 100
        || *(_QWORD *)a1 == 0x666F727070696B73LL && *(_WORD *)(a1 + 8) == 27753 && *(_BYTE *)(a1 + 10) == 101
        || *(_QWORD *)a1 == 0x74746E656D656C65LL && *(_WORD *)(a1 + 8) == 28793 && *(_BYTE *)(a1 + 10) == 101
        || *(_QWORD *)a1 == 0x696C616974696E69LL && *(_WORD *)(a1 + 8) == 25978 && *(_BYTE *)(a1 + 10) == 115 )
      {
        return 1;
      }
      break;
    case 17LL:
      if ( !(*(_QWORD *)a1 ^ 0x6974696E61736F6ELL | *(_QWORD *)(a1 + 8) ^ 0x646E756F625F657ALL)
        && *(_BYTE *)(a1 + 16) == 115
        || !(*(_QWORD *)a1 ^ 0x74696C7073657270LL | *(_QWORD *)(a1 + 8) ^ 0x6E6974756F726F63LL)
        && *(_BYTE *)(a1 + 16) == 101
        || !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x6D69746C6165725FLL)
        && *(_BYTE *)(a1 + 16) == 101 )
      {
        return 1;
      }
      break;
    case 21LL:
      if ( !(*(_QWORD *)a1 ^ 0x696F705F6C6C756ELL | *(_QWORD *)(a1 + 8) ^ 0x5F73695F7265746ELL)
        && *(_DWORD *)(a1 + 16) == 1768710518
        && *(_BYTE *)(a1 + 20) == 100
        || !(*(_QWORD *)a1 ^ 0x6E696C6E692D6F6ELL | *(_QWORD *)(a1 + 8) ^ 0x742D656E696C2D65LL)
        && *(_DWORD *)(a1 + 16) == 1701601889
        && *(_BYTE *)(a1 + 20) == 115 )
      {
        return 1;
      }
      break;
    case 13LL:
      if ( *(_QWORD *)a1 == 0x7566726F6674706FLL && *(_DWORD *)(a1 + 8) == 1852406394 && *(_BYTE *)(a1 + 12) == 103
        || *(_QWORD *)a1 == 0x5F736E7275746572LL && *(_DWORD *)(a1 + 8) == 1667856244 && *(_BYTE *)(a1 + 12) == 101
        || *(_QWORD *)a1 == 0x657A6974696E6173LL && *(_DWORD *)(a1 + 8) == 1887007839 && *(_BYTE *)(a1 + 12) == 101 )
      {
        return 1;
      }
      break;
    case 28LL:
      if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x636972656D756E5FLL)
        && *(_QWORD *)(a1 + 16) == 0x69626174735F6C61LL
        && *(_DWORD *)(a1 + 24) == 2037672300 )
      {
        return 1;
      }
      break;
    case 26LL:
      if ( !(*(_QWORD *)a1 ^ 0x657A6974696E6173LL | *(_QWORD *)(a1 + 8) ^ 0x6D69746C6165725FLL)
        && *(_QWORD *)(a1 + 16) == 0x696B636F6C625F65LL
        && *(_WORD *)(a1 + 24) == 26478
        || !(*(_QWORD *)a1 ^ 0x74616C7563657073LL | *(_QWORD *)(a1 + 8) ^ 0x64616F6C5F657669LL)
        && *(_QWORD *)(a1 + 16) == 0x696E65647261685FLL
        && *(_WORD *)(a1 + 24) == 26478 )
      {
        return 1;
      }
      break;
    case 23LL:
      if ( !(*(_QWORD *)a1 ^ 0x6572656665726564LL | *(_QWORD *)(a1 + 8) ^ 0x5F656C626165636ELL)
        && *(_DWORD *)(a1 + 16) == 1851748975
        && *(_WORD *)(a1 + 20) == 27765
        && *(_BYTE *)(a1 + 22) == 108
        || !(*(_QWORD *)a1 ^ 0x656E6769732D6F6ELL | *(_QWORD *)(a1 + 8) ^ 0x2D736F72657A2D64LL)
        && *(_DWORD *)(a1 + 16) == 1831694438
        && *(_WORD *)(a1 + 20) == 29793
        && *(_BYTE *)(a1 + 22) == 104
        || !(*(_QWORD *)a1 ^ 0x2D656C69666F7270LL | *(_QWORD *)(a1 + 8) ^ 0x612D656C706D6173LL)
        && *(_DWORD *)(a1 + 16) == 1920295779
        && *(_WORD *)(a1 + 20) == 29793
        && *(_BYTE *)(a1 + 22) == 101 )
      {
        return 1;
      }
      break;
    default:
      if ( a2 == 20
        && !(*(_QWORD *)a1 ^ 0x6C616D726F6E6564LL | *(_QWORD *)(a1 + 8) ^ 0x6874616D2D70662DLL)
        && *(_DWORD *)(a1 + 16) == 842229293 )
      {
        return 1;
      }
      break;
  }
  return 0;
}
